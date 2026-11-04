import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Sequence, Tuple

def compute_basic_summary(series: np.ndarray) -> Dict[str, float]:
    """Computes basic descriptive summary statistics (mean, std, min, max, median)."""
    if len(series) == 0:
        return {}
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "median": float(np.median(series))
    }


# =============================================================================
# Statistical Reality Check
# =============================================================================
# All numbers below are computed from the supplied data; nothing is tabulated or
# hardcoded. Conventions:
#   rho(t)     normalised autocorrelation function (biased estimator, FFT).
#   tau_int    integrated autocorrelation time, Sokal convention
#              tau_int = 1/2 + sum_{t=1..M} rho(t); white noise gives 1/2.
#   g          statistical inefficiency = 2 * tau_int (>= 1 enforced).
#   ESS        effective sample size = N / g (<= N).
#   SEM_corr   s / sqrt(ESS), with s the sample standard deviation (ddof=1).
# Window M: Sokal's automatic windowing, smallest M with M >= c * tau_int(M).
# Uncertainty of tau_int: Madras & Sokal approximation
#   var(tau) ~ 2 (2M + 1) / N * tau^2.
# Statuses: "ok" | "inconclusive" | "insufficient_data" (never "proven").
# =============================================================================

MIN_SAMPLES = 20          # below this no statistical diagnostic is attempted
SOKAL_C = 5.0             # windowing constant
MIN_TAU_RATIO = 20.0      # N / tau_int below this -> tau_int estimate inconclusive
LOW_ESS = 20.0            # ESS below this is flagged as too small to trust
MAX_TAU_REL_ERR = 0.3     # relative std. error of tau_int above this -> inconclusive

LIMITATIONS = [
    "All diagnostics use a single trajectory; they cannot detect slow processes that never occur in it "
    "(e.g. a transition that happens once, or not at all, in the simulated time).",
    "Autocorrelation/ESS assume the analysed segment is stationary; any residual drift biases tau_int, ESS "
    "and the standard error.",
    "Candidate equilibration detection finds the cut that maximises the effective number of samples; it "
    "does not prove equilibrium and can be misled by slow drifts or by noisy series.",
    "Block-averaging uncertainties are themselves noisy when few blocks remain (relative error of a block "
    "SEM is about 1/sqrt(2(n_blocks-1))).",
    "The series is treated as equally spaced samples; frames must be uniformly spaced in time.",
    "A statistic of one observable (e.g. RMSD) says nothing about the convergence of others.",
]
RMSF_LIMITATIONS = [
    "Window-to-window agreement of RMSF profiles shows reproducibility between those windows only; it does "
    "not prove that conformational sampling is converged.",
    "RMSF grows with window length, so profiles from windows of different duration are not directly comparable.",
    "Profiles from overlapping windows are not independent and overstate agreement.",
    "Status thresholds (Pearson r, relative RMSE, CV) are heuristic and adjustable; they are not statistical tests.",
    "RMSF from gmx rmsf has no PBC handling: periodic jumps in any window distort that window's profile.",
]


def _f(x: Any) -> Optional[float]:
    """float or None (NaN/inf -> None) so results are JSON-safe."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _as_series(x: Any, name: str = "series") -> np.ndarray:
    arr = np.asarray(x, dtype=float) if not isinstance(x, pd.Series) else x.to_numpy(dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/inf values; clean or remove them before analysis.")
    return arr


def infer_timestep(times: Sequence[float], rtol: float = 1e-3) -> Dict[str, Any]:
    """Frame spacing from a time column. uniform=False means the series is not equally spaced
    and the diagnostics below must not be applied without resampling."""
    t = _as_series(times, "times")
    if len(t) < 2:
        return {"dt": None, "uniform": False, "reason": "fewer than 2 time points"}
    d = np.diff(t)
    dt = float(np.median(d))
    if dt <= 0:
        return {"dt": None, "uniform": False, "reason": "time is not strictly increasing"}
    uniform = bool(np.all(np.abs(d - dt) <= rtol * abs(dt)))
    return {"dt": dt, "uniform": uniform,
            "reason": None if uniform else "frame spacing varies by more than the tolerance"}


# ------------------------------------------------------------ autocorrelation
def autocorrelation_function(x: Any, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalised ACF rho(0..max_lag) (rho(0)=1) via FFT. Raises on zero variance."""
    x = _as_series(x)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 samples for an autocorrelation function.")
    if np.ptp(x) == 0:
        raise ValueError("Series has zero variance; autocorrelation is undefined.")
    xc = x - x.mean()
    f = np.fft.rfft(xc, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0 or not np.isfinite(acf[0]):
        raise ValueError("Series has zero variance; autocorrelation is undefined.")
    acf = acf / acf[0]
    if max_lag is not None:
        acf = acf[: int(max_lag) + 1]
    return acf


def integrated_autocorrelation_time(x: Any, dt: Optional[float] = None, time_unit: str = "",
                                    c: float = SOKAL_C, max_lag: Optional[int] = None,
                                    include_curve: bool = False,
                                    min_samples: int = MIN_SAMPLES) -> Dict[str, Any]:
    """Integrated autocorrelation time with Sokal automatic windowing."""
    x = _as_series(x)
    n = len(x)
    out: Dict[str, Any] = {"n_samples": n, "convention": "tau_int = 1/2 + sum rho(t) (white noise = 0.5)",
                           "window_constant_c": c, "time_unit": time_unit, "warnings": []}
    base = {"status": "insufficient_data", "tau_int_frames": None, "tau_int_stderr_frames": None,
            "tau_int_time": None, "window_frames": None, "statistical_inefficiency": None,
            "n_over_tau": None}
    if n < min_samples:
        out.update(base)
        out["warnings"].append(f"Only {n} samples (< {min_samples}); autocorrelation not attempted.")
        return out
    try:
        rho = autocorrelation_function(x, max_lag if max_lag is not None else n // 2)
    except ValueError as exc:
        out.update(base)
        out["status"] = "inconclusive"
        out["warnings"].append(str(exc))
        return out
    tau_cum = np.cumsum(rho) - 0.5
    lags = np.arange(len(rho))
    ok = np.nonzero((lags >= 1) & (lags >= c * tau_cum))[0]
    window_found = len(ok) > 0
    M = int(ok[0]) if window_found else len(rho) - 1
    tau = float(max(tau_cum[M], 0.5))
    if tau_cum[M] < 0.5:
        out["warnings"].append("Estimated tau_int < 1/2 (negative autocorrelation); clamped to 1/2 "
                               "(statistical inefficiency 1).")
    stderr = tau * math.sqrt(2.0 * (2 * M + 1) / n)
    ratio = n / tau
    status = "ok"
    if not window_found:
        status = "inconclusive"
        out["warnings"].append("Autocorrelation window did not converge within the available lags: the series "
                               "is too short relative to its correlation time; tau_int is a lower bound.")
    elif ratio < MIN_TAU_RATIO:
        status = "inconclusive"
        out["warnings"].append(f"Series spans only {ratio:.1f} integrated autocorrelation times "
                               f"(< {MIN_TAU_RATIO:g}); tau_int is unreliable.")
    rel_err = stderr / tau
    if status == "ok" and rel_err > MAX_TAU_REL_ERR:
        status = "inconclusive"
        out["warnings"].append(f"Relative standard error of tau_int is {rel_err:.0%} (> {MAX_TAU_REL_ERR:.0%}): "
                               "too noisy to trust; a short series can underestimate slow correlations.")
    out.update({"status": status, "tau_int_frames": tau, "tau_int_stderr_frames": float(stderr),
                "tau_int_time": _f(tau * dt) if dt else None, "window_frames": M,
                "statistical_inefficiency": float(max(2.0 * tau, 1.0)), "n_over_tau": float(ratio)})
    if include_curve:
        keep = min(len(rho), max(100, 3 * M + 1))
        out["acf"] = {"lags_frames": list(range(keep)), "values": [float(v) for v in rho[:keep]]}
    return out


# ---------------------------------------------------------------- ESS and SEM
def effective_sample_size(x: Any, dt: Optional[float] = None, time_unit: str = "",
                          value_unit: str = "", min_samples: int = MIN_SAMPLES) -> Dict[str, Any]:
    """ESS and correlation-aware standard error of the mean (value_unit)."""
    x = _as_series(x)
    n = len(x)
    ti = integrated_autocorrelation_time(x, dt=dt, time_unit=time_unit, min_samples=min_samples)
    res: Dict[str, Any] = {
        "n_samples": n, "value_unit": value_unit, "mean": _f(x.mean()) if n else None,
        "std": _f(x.std(ddof=1)) if n > 1 else None,
        "sem_naive": _f(x.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
        "tau_int_frames": ti["tau_int_frames"], "tau_int_time": ti["tau_int_time"], "time_unit": time_unit,
        "statistical_inefficiency": ti["statistical_inefficiency"], "ess": None, "sem_corrected": None,
        "sem_inflation_factor": None, "status": ti["status"], "warnings": list(ti["warnings"]),
    }
    if ti["tau_int_frames"] is None:
        return res
    g = ti["statistical_inefficiency"]
    ess = n / g
    res["ess"] = float(ess)
    res["sem_corrected"] = _f(res["std"] / math.sqrt(ess))
    res["sem_inflation_factor"] = float(math.sqrt(g))
    # relative uncertainty of the SEM follows from that of tau_int
    res["sem_relative_uncertainty"] = _f(0.5 * ti["tau_int_stderr_frames"] / ti["tau_int_frames"])
    if ess < LOW_ESS:
        res["warnings"].append(f"Effective sample size {ess:.1f} < {LOW_ESS:g}: mean and its error bar are "
                               "poorly determined.")
        if res["status"] == "ok":
            res["status"] = "inconclusive"
    return res


# --------------------------------------------------------------- equilibration
def _g_of(x: np.ndarray) -> float:
    ti = integrated_autocorrelation_time(x, min_samples=4)
    return ti["statistical_inefficiency"] if ti["statistical_inefficiency"] is not None else float("nan")


def _detect_equilibration_native(x: np.ndarray, nskip: int, max_frac: float) -> Tuple[int, float, float]:
    """Chodera (2016) criterion: choose t0 maximising N_eff(t0) = (N - t0) / g(t0), g from the
    Sokal tau_int of x[t0:]. Only cuts up to max_frac*N are considered."""
    n = len(x)
    t_max = int(n * max_frac)
    best = (0, float("nan"), -np.inf)
    for t0 in range(0, t_max + 1, nskip):
        g = _g_of(x[t0:])
        if not np.isfinite(g):
            continue
        neff = (n - t0) / g
        if neff > best[2]:
            best = (t0, g, neff)
    if not np.isfinite(best[1]):
        raise ValueError("Could not evaluate any equilibration candidate (zero variance?).")
    return best


def _detect_equilibration_pymbar(x: np.ndarray, nskip: int) -> Tuple[int, float, float]:
    from pymbar import timeseries  # optional dependency
    t0, g, neff = timeseries.detect_equilibration(x, fast=True, nskip=nskip)
    return int(t0), float(g), float(neff)


def _timestamp(index: int, dt: Optional[float], t_start: Optional[float],
               times: Optional[np.ndarray]) -> Optional[float]:
    """Simulation time of series frame `index`: the supplied timestamp if `times` is given, else
    t_start + index*dt (t_start None -> 0, i.e. time elapsed since the first frame of the series)."""
    if times is not None:
        return _f(times[index])
    if not dt:
        return None
    return _f((t_start or 0.0) + index * dt)


def _check_times(times: Optional[Sequence[float]], n: int) -> Optional[np.ndarray]:
    if times is None:
        return None
    t = _as_series(times, "times")
    if len(t) != n:
        raise ValueError(f"times has {len(t)} entries but the series has {n} frames.")
    if n > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing.")
    return t


def detect_equilibration(x: Any, dt: Optional[float] = None, time_unit: str = "", method: str = "auto",
                         nskip: Optional[int] = None, max_discard_fraction: float = 0.5,
                         min_samples: int = 50, t_start: Optional[float] = None,
                         times: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Data-driven CANDIDATE equilibration time (never a proof).

    method: 'pymbar' -> pymbar.timeseries.detect_equilibration (Chodera, JCTC 2016, 12, 1799);
            'native' -> same criterion with this module's tau_int estimator (see _detect_equilibration_native);
            'auto'   -> pymbar if importable, otherwise native (recorded in 'method' and 'warnings').
    Status 'candidate' means a cut was found and passed the sanity checks; 'inconclusive' means it did
    not (cut at the search limit, or residual drift between halves of the remaining data).

    Time reporting: 't0_frame' is the index within the series. 't0_time' is the simulation timestamp of
    that frame: times[t0] if `times` is supplied, else t_start + t0*dt. 't0_elapsed' = t0*dt is the time
    since the first frame of the series. 'time_origin' says where the timestamp came from; if neither
    t_start nor times is given the first frame is assumed to be at time 0 ('assumed_zero').
    """
    if method not in ("auto", "pymbar", "native"):
        raise ValueError("method must be 'auto', 'pymbar' or 'native'.")
    if not (0 < max_discard_fraction < 1):
        raise ValueError("max_discard_fraction must be in (0, 1).")
    x = _as_series(x)
    n = len(x)
    tarr = _check_times(times, n)
    origin = "supplied_times" if tarr is not None else ("supplied_t_start" if t_start is not None else "assumed_zero")
    res: Dict[str, Any] = {
        "n_samples": n, "method": None, "status": "insufficient_data", "t0_frame": None, "t0_time": None,
        "t0_elapsed": None, "time_origin": origin, "series_start_time": _timestamp(0, dt, t_start, tarr) if n else None,
        "time_unit": time_unit, "n_discarded": None, "fraction_discarded": None, "n_remaining": None,
        "statistical_inefficiency_at_t0": None, "ess_at_t0": None, "drift_z_score": None, "warnings": [],
        "interpretation": "Candidate only: the data cannot prove that the system is equilibrated.",
    }
    if n < min_samples:
        res["warnings"].append(f"Only {n} samples (< {min_samples}); equilibration detection not attempted.")
        return res
    if np.ptp(x) == 0:
        res["status"] = "inconclusive"
        res["method"] = "none"
        res["warnings"].append("Series is constant; equilibration cannot be assessed.")
        return res
    nskip = int(nskip) if nskip else max(1, n // 100)

    used = method
    result = None
    if method in ("auto", "pymbar"):
        try:
            result = _detect_equilibration_pymbar(x, nskip)
            used = "pymbar"
        except ImportError:
            if method == "pymbar":
                raise
            res["warnings"].append("pymbar is not installed; used the built-in implementation of the same "
                                   "criterion (results can differ slightly).")
        except Exception as exc:  # pymbar failed on this data: report, do not hide
            if method == "pymbar":
                raise
            res["warnings"].append(f"pymbar failed ({type(exc).__name__}: {exc}); used built-in implementation.")
    if result is None:
        result = _detect_equilibration_native(x, nskip, max_discard_fraction)
        used = "native"
    t0, g, neff = result
    res.update({"method": used, "t0_frame": int(t0), "t0_time": _timestamp(int(t0), dt, t_start, tarr),
                "t0_elapsed": _f(t0 * dt) if dt else None,
                "n_discarded": int(t0), "fraction_discarded": float(t0 / n), "n_remaining": int(n - t0),
                "statistical_inefficiency_at_t0": _f(g), "ess_at_t0": _f(neff), "nskip": nskip})
    status = "candidate"
    if t0 > max_discard_fraction * n - nskip:
        status = "inconclusive"
        res["warnings"].append(f"Candidate cut ({t0 / n:.0%} of the series) is at the search limit "
                               f"({max_discard_fraction:.0%}): the series may not reach a stationary regime.")
    rest = x[t0:]
    h = len(rest) // 2
    if h >= 20:
        a, b = rest[:h], rest[h:2 * h]
        ea, eb = effective_sample_size(a), effective_sample_size(b)
        se = [e["sem_corrected"] for e in (ea, eb)]
        if all(s is not None for s in se) and (se[0] ** 2 + se[1] ** 2) > 0:
            z = abs(a.mean() - b.mean()) / math.sqrt(se[0] ** 2 + se[1] ** 2)
            res["drift_z_score"] = float(z)
            if z > 3.0:
                status = "inconclusive"
                res["warnings"].append(f"Means of the two halves after the candidate cut differ by {z:.1f} "
                                       "standard errors: residual drift, not stationary.")
    else:
        res["warnings"].append("Too few samples after the cut to test for residual drift.")
    res["status"] = status
    return res


# ------------------------------------------------------------ block averaging
def block_averaging(x: Any, block_sizes: Optional[Sequence[int]] = None, dt: Optional[float] = None,
                    time_unit: str = "", value_unit: str = "", min_blocks: int = 4,
                    plateau_min_blocks: int = 16, n_plateau: int = 3,
                    min_samples: int = MIN_SAMPLES) -> Dict[str, Any]:
    """Flyvbjerg-Petersen block averaging over several block sizes.

    For block size b: n_b = N // b blocks (trailing remainder discarded and reported),
    SEM(b) = std(block means, ddof=1) / sqrt(n_b), with its own uncertainty SEM / sqrt(2(n_b-1)).
    Convergence: among block sizes with >= plateau_min_blocks blocks, the largest n_plateau sizes must
    be mutually consistent (within 2 combined uncertainties) and not show a significant rising trend.
    """
    x = _as_series(x)
    n = len(x)
    res: Dict[str, Any] = {"n_samples": n, "value_unit": value_unit, "time_unit": time_unit, "blocks": [],
                           "convergence": {"status": "insufficient_data"}, "warnings": []}
    if n < min_samples:
        res["warnings"].append(f"Only {n} samples (< {min_samples}); block averaging not attempted.")
        return res
    if block_sizes is None:
        sizes, b = [], 1
        while n // b >= min_blocks:
            sizes.append(b)
            b *= 2
    else:
        sizes = sorted({int(b) for b in block_sizes})
        if any(b < 1 for b in sizes):
            raise ValueError("Block sizes must be >= 1.")
        too_big = [b for b in sizes if n // b < 2]
        if too_big:
            raise ValueError(f"Block sizes {too_big} leave fewer than 2 blocks for N={n}.")
    if np.ptp(x) == 0:
        res["warnings"].append("Series is constant; block statistics are degenerate.")
        res["convergence"] = {"status": "inconclusive", "reason": "zero variance"}
        return res
    rows = []
    for b in sizes:
        nb = n // b
        means = x[: nb * b].reshape(nb, b).mean(axis=1)
        sem = float(means.std(ddof=1) / math.sqrt(nb))
        rows.append({"block_size_frames": b, "block_size_time": _f(b * dt) if dt else None, "n_blocks": nb,
                     "n_discarded_frames": n - nb * b, "sem": sem,
                     "sem_uncertainty": float(sem / math.sqrt(2.0 * (nb - 1)))})
    res["blocks"] = rows
    sem_naive = float(x.std(ddof=1) / math.sqrt(n))
    res["sem_naive"] = sem_naive
    elig = [r for r in rows if r["n_blocks"] >= plateau_min_blocks]
    if len(elig) < n_plateau:
        res["convergence"] = {"status": "insufficient_data",
                              "reason": f"fewer than {n_plateau} block sizes have >= {plateau_min_blocks} blocks"}
        res["warnings"].append("Not enough data to judge a block-size plateau.")
        return res
    tail = elig[-n_plateau:]
    consistent = all(abs(p["sem"] - q["sem"]) <= 2.0 * math.sqrt(p["sem_uncertainty"] ** 2 + q["sem_uncertainty"] ** 2)
                     for i, p in enumerate(tail) for q in tail[i + 1:])
    first, last = tail[0], tail[-1]
    rising = all(tail[i]["sem"] < tail[i + 1]["sem"] for i in range(len(tail) - 1)) and \
        (last["sem"] - first["sem"]) > 2.0 * math.sqrt(first["sem_uncertainty"] ** 2 + last["sem_uncertainty"] ** 2)
    plateau_sem = float(np.mean([r["sem"] for r in tail]))
    conv = {"plateau_block_sizes_frames": [r["block_size_frames"] for r in tail],
            "plateau_min_blocks": plateau_min_blocks, "sem_at_largest_eligible": last["sem"]}
    if consistent and not rising:
        conv.update({"status": "converged", "sem_plateau": plateau_sem,
                     "sem_plateau_uncertainty": float(np.mean([r["sem_uncertainty"] for r in tail]) /
                                                      math.sqrt(len(tail))),
                     "implied_statistical_inefficiency": float(max((plateau_sem / sem_naive) ** 2, 1.0))})
        conv["implied_ess"] = float(n / conv["implied_statistical_inefficiency"])
    else:
        conv.update({"status": "not_converged", "sem_plateau": None,
                     "reason": ("SEM still rising with block size" if rising else
                                "SEM values at the largest block sizes are inconsistent")})
        res["warnings"].append("Block SEM has not plateaued: the correlation time is comparable to the block "
                               "sizes available, so any SEM from this series is likely an underestimate.")
    res["convergence"] = conv
    return res


# -------------------------------------------------------------- orchestrator
def statistical_reality_check(series: Any, dt: Optional[float] = None, time_unit: str = "ns",
                              value_unit: str = "", equilibration_method: str = "auto",
                              include_acf_curve: bool = False, t_start: Optional[float] = None,
                              times: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Runs autocorrelation, ESS/SEM, candidate equilibration and block averaging on one observable.

    ACF/ESS/block statistics are computed on the segment after the candidate equilibration time when
    the candidate passes its checks, otherwise on the full series (with a warning). dt is the frame
    spacing in time_unit; value_unit is the unit of the observable (e.g. 'nm'). Reported times are
    simulation timestamps: pass `times` (one per frame) or `t_start` (time of the first frame) for
    trajectories that do not start at 0; otherwise the first frame is assumed to be at time 0.
    """
    x = _as_series(series)
    n = len(x)
    tarr = _check_times(times, n)
    if dt is not None and not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be a positive number, got {dt!r}.")
    warnings: List[str] = []
    out: Dict[str, Any] = {
        "inputs": {"n_frames": n, "dt": dt, "time_unit": time_unit, "value_unit": value_unit,
                   "duration": _f((n - 1) * dt) if dt and n else None,
                   "start_time": _timestamp(0, dt, t_start, tarr) if n else None,
                   "end_time": _timestamp(n - 1, dt, t_start, tarr) if n else None},
        "limitations": list(LIMITATIONS),
    }
    eq = detect_equilibration(x, dt=dt, time_unit=time_unit, method=equilibration_method,
                              t_start=t_start, times=tarr)
    out["equilibration"] = eq
    if eq["status"] == "candidate":
        start, basis = eq["t0_frame"], "after candidate equilibration time"
    else:
        start, basis = 0, "full series"
        if eq["status"] == "inconclusive":
            warnings.append("No usable equilibration candidate: statistics use the full series, which may "
                            "include a non-stationary transient and therefore be biased.")
    seg = x[start:]
    out["analysis_segment"] = {"basis": basis, "start_frame": int(start), "n_frames": int(len(seg)),
                               "start_time": _timestamp(int(start), dt, t_start, tarr), "time_unit": time_unit}
    out["autocorrelation"] = integrated_autocorrelation_time(seg, dt=dt, time_unit=time_unit,
                                                            include_curve=include_acf_curve)
    out["effective_sample_size"] = effective_sample_size(seg, dt=dt, time_unit=time_unit, value_unit=value_unit)
    out["block_averaging"] = block_averaging(seg, dt=dt, time_unit=time_unit, value_unit=value_unit)

    reasons: List[str] = []
    ess_acf = out["effective_sample_size"]["ess"]
    conv = out["block_averaging"]["convergence"]
    cross: Dict[str, Any] = {"ess_acf": ess_acf, "ess_block": conv.get("implied_ess"), "agreement": None}
    if ess_acf is not None and conv.get("implied_ess"):
        ratio = max(ess_acf, conv["implied_ess"]) / min(ess_acf, conv["implied_ess"])
        cross["ratio"] = float(ratio)
        cross["agreement"] = bool(ratio <= 2.0)
        if not cross["agreement"]:
            reasons.append(f"ESS from autocorrelation ({ess_acf:.1f}) and from block averaging "
                           f"({conv['implied_ess']:.1f}) differ by more than a factor of 2.")
    out["cross_checks"] = cross

    if n < MIN_SAMPLES:
        status = "insufficient_data"
        reasons.append(f"Only {n} frames.")
    else:
        if eq["status"] != "candidate":
            reasons.append(f"Equilibration: {eq['status']}.")
        for key, label in (("autocorrelation", "Autocorrelation"), ("effective_sample_size", "ESS")):
            if out[key]["status"] != "ok":
                reasons.append(f"{label}: {out[key]['status']}.")
        if conv["status"] != "converged":
            reasons.append(f"Block averaging: {conv['status']}.")
        status = "no_issues_detected" if not reasons else "inconclusive"
    for sec in ("equilibration", "autocorrelation", "effective_sample_size", "block_averaging"):
        for w in out[sec].get("warnings", []):
            if w not in warnings:
                warnings.append(w)
    out["overall"] = {
        "status": status, "reasons": reasons,
        "meaning": ("'no_issues_detected' means these diagnostics found no problem in this series; it is not "
                    "proof of equilibration or convergence. 'inconclusive' means at least one diagnostic "
                    "could not support a reliable error estimate."),
    }
    out["warnings"] = warnings
    return out


# -------------------------------------------------- RMSF temporal reliability
def plan_time_windows(begin: float, end: float, n_windows: int = 2) -> List[Tuple[float, float]]:
    """Equal-length, contiguous, non-overlapping windows covering [begin, end] (any time unit).
    Use each as -b/-e of a SEPARATE gmx rmsf run; RMSF is never split by residue."""
    if n_windows < 2:
        raise ValueError("Need at least 2 windows.")
    if not (np.isfinite(begin) and np.isfinite(end)) or end <= begin or begin < 0:
        raise ValueError("Need 0 <= begin < end.")
    step = (end - begin) / n_windows
    return [(begin + i * step, begin + (i + 1) * step) for i in range(n_windows)]


def _profile(p: Any, i: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ids = None
    if isinstance(p, pd.DataFrame):
        if "RMSF" not in p.columns:
            raise ValueError(f"Profile {i}: DataFrame needs an 'RMSF' column (as produced by run_rmsf).")
        for key in ("Residue", "Atom"):
            if key in p.columns:
                ids = p[key].to_numpy()
                break
        vals = p["RMSF"].to_numpy(dtype=float)
    else:
        vals = _as_series(p, f"profile {i}")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"Profile {i} contains NaN/inf.")
    if np.any(vals < 0):
        raise ValueError(f"Profile {i} contains negative RMSF values.")
    return np.asarray(vals, dtype=float), ids


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def compare_rmsf_windows(profiles: Sequence[Any], labels: Optional[Sequence[str]] = None,
                         windows: Optional[Sequence[Tuple[float, float]]] = None,
                         window_unit: str = "ns", residue_ids: Optional[Sequence[Any]] = None,
                         rmsf_unit: str = "nm", consistent_r: float = 0.9, moderate_r: float = 0.7,
                         max_rel_rmse: float = 0.25, flag_cv: float = 0.5,
                         top_fraction: float = 0.1) -> Dict[str, Any]:
    """Compare residue-wise RMSF profiles computed from SEPARATE trajectory time windows.

    profiles: >= 2 arrays (or run_rmsf DataFrames) of identical length/order, one per window.
    windows : optional (begin, end) per profile, used to check equal duration and non-overlap.
    Residues are never split into pseudo time blocks; a single profile yields 'insufficient_data'.
    Status: consistent / partially_consistent / inconsistent (heuristic thresholds, reported in
    'thresholds'), inconclusive (undefined correlation or overlapping windows) or insufficient_data.
    """
    res: Dict[str, Any] = {"n_windows": len(profiles), "rmsf_unit": rmsf_unit, "window_unit": window_unit,
                           "status": "insufficient_data", "warnings": [], "limitations": list(RMSF_LIMITATIONS),
                           "thresholds": {"consistent_min_pearson": consistent_r,
                                          "partial_min_pearson": moderate_r, "max_relative_rmse": max_rel_rmse,
                                          "flag_cv": flag_cv, "top_fraction": top_fraction}}
    if len(profiles) < 2:
        res["warnings"].append("At least two RMSF profiles from separate time windows are required; the "
                               "residues of one profile cannot be treated as time blocks.")
        return res
    labels = list(labels) if labels is not None else [f"window_{i + 1}" for i in range(len(profiles))]
    if len(labels) != len(profiles):
        raise ValueError("labels must match the number of profiles.")
    parsed = [_profile(p, i) for i, p in enumerate(profiles)]
    n_res = len(parsed[0][0])
    if any(len(v) != n_res for v, _ in parsed):
        raise ValueError(f"RMSF profiles differ in length {[len(v) for v, _ in parsed]}; they must cover the "
                         "same residues/atoms in the same order.")
    id_sets = [ids for _, ids in parsed if ids is not None]
    if residue_ids is not None:
        ids = np.asarray(residue_ids)
        if len(ids) != n_res:
            raise ValueError("residue_ids length does not match the profiles.")
    elif id_sets:
        if len(id_sets) != len(parsed) or any(not np.array_equal(id_sets[0], s) for s in id_sets[1:]):
            raise ValueError("Residue/atom identifiers differ between profiles (or are missing from some).")
        ids = id_sets[0]
    else:
        ids = np.arange(1, n_res + 1)
        res["warnings"].append("No residue identifiers supplied; positions 1..N used.")
    if n_res < 10:
        res["warnings"].append(f"Only {n_res} residues/atoms; profile comparison is not meaningful.")
        return res
    mat = np.vstack([v for v, _ in parsed])  # windows x residues

    independent = True
    if windows is not None:
        if len(windows) != len(profiles):
            raise ValueError("windows must match the number of profiles.")
        w = [(float(b), float(e)) for b, e in windows]
        if any(e <= b for b, e in w):
            raise ValueError("Each window needs end > begin.")
        dur = np.array([e - b for b, e in w])
        res["windows"] = [{"label": l, "begin": b, "end": e, "duration": e - b} for l, (b, e) in zip(labels, w)]
        if dur.max() / dur.min() > 1.1:
            res["warnings"].append("Windows differ in duration by more than 10%; RMSF grows with window length, "
                                   "so differences partly reflect duration, not reproducibility.")
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if max(w[i][0], w[j][0]) < min(w[i][1], w[j][1]):
                    independent = False
        if not independent:
            res["warnings"].append("Some windows overlap: profiles are not independent and agreement is "
                                   "overstated.")
    else:
        res["warnings"].append("No window boundaries supplied; overlap and duration mismatch were not checked.")
    res["independent_windows"] = independent

    overall_mean = float(mat.mean())
    pairs, undefined = [], False
    k = max(1, int(round(top_fraction * n_res)))
    tops = [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in mat]
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            r = _pearson(mat[i], mat[j])
            undefined |= r is None
            rmse = float(np.sqrt(np.mean((mat[i] - mat[j]) ** 2)))
            rs = _pearson(pd.Series(mat[i]).rank().to_numpy(), pd.Series(mat[j]).rank().to_numpy())
            pairs.append({"a": labels[i], "b": labels[j], "pearson": r, "spearman": rs, "rmse": rmse,
                          "mean_abs_diff": float(np.mean(np.abs(mat[i] - mat[j]))),
                          "relative_rmse": _f(rmse / overall_mean) if overall_mean > 0 else None,
                          "top_flexible_jaccard": float(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))})
    res["pairwise"] = pairs
    mean = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1)
    cv = np.where(mean > 0, sd / np.where(mean > 0, mean, 1), np.nan)
    flagged = np.nonzero(np.nan_to_num(cv, nan=0.0) > flag_cv)[0]
    res["per_residue"] = {"ids": [_py(i) for i in ids],
                          "mean": [float(v) for v in mean], "std": [float(v) for v in sd],
                          "cv": [_f(v) for v in cv], "range": [float(v) for v in np.ptp(mat, axis=0)]}
    res["unstable_residues"] = {"cv_threshold": flag_cv, "count": int(len(flagged)),
                                "fraction": float(len(flagged) / n_res),
                                "ids": [_py(ids[i]) for i in flagged[np.argsort(-cv[flagged])][:25]]}
    rs_ok = [p["pearson"] for p in pairs if p["pearson"] is not None]
    rel = [p["relative_rmse"] for p in pairs if p["relative_rmse"] is not None]
    res["summary"] = {
        "min_pearson": _f(min(rs_ok)) if rs_ok else None,
        "median_pearson": _f(np.median(rs_ok)) if rs_ok else None,
        "max_relative_rmse": _f(max(rel)) if rel else None,
        "mean_top_flexible_jaccard": float(np.mean([p["top_flexible_jaccard"] for p in pairs])),
    }
    if undefined or not rs_ok:
        res["status"] = "inconclusive"
        res["warnings"].append("A profile is constant, so its correlation is undefined.")
    else:
        mp, rr = res["summary"]["min_pearson"], res["summary"]["max_relative_rmse"]
        if mp >= consistent_r and rr is not None and rr <= max_rel_rmse:
            res["status"] = "consistent"
        elif mp >= moderate_r:
            res["status"] = "partially_consistent"
        else:
            res["status"] = "inconsistent"
        if not independent and res["status"] != "inconsistent":
            res["status"] = "inconclusive"
            res["warnings"].append("Status downgraded to inconclusive because windows overlap.")
    return res
