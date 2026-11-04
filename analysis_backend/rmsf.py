import os
import subprocess
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .analysis_utils import AnalysisResult, validate_inputs, validate_group_selection, parse_xvg, get_gromacs_version_info, remove_stale_outputs
from .statistics import compute_basic_summary, compare_rmsf_windows, plan_time_windows
from .plotting import get_xmgrace_command, launch_xmgrace
from .gmx_options import require_bool, normalize_time_window, time_args, resolve_group, resolve_group_ref, require_finite


MAX_RELIABILITY_WINDOWS = 20


def _execute_rmsf(cmd: List[str], group_input: str) -> None:
    """Run one gmx rmsf command (group answered on stdin). Raises like the original inline code."""
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=group_input)
        if process.returncode != 0:
            raise RuntimeError(f"gmx rmsf execution FAILED:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    except FileNotFoundError:
        error_msg = "GROMACS binary 'gmx' not found on PATH for gmx rmsf."
        print(f"[RMSF] ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)


def _not_run(code: str, reason: str, status: str = "not_run", **extra: Any) -> Dict[str, Any]:
    out = {"status": status, "reason_code": code, "reason": reason}
    out.update(extra)
    return out


def _per_residue_differences(ids: Sequence[Any], profiles: List[np.ndarray], labels: List[str],
                             top: int = 10) -> List[Dict[str, Any]]:
    """Signed per-residue differences (later window minus earlier window, nm) for every window pair."""
    out = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            d = profiles[j] - profiles[i]
            order = np.argsort(-np.abs(d), kind="stable")[:top]
            out.append({
                "a": labels[i], "b": labels[j], "difference": "b minus a", "unit": "nm",
                "ids": [x.item() if hasattr(x, "item") else x for x in ids],
                "values": [float(v) for v in d],
                "largest_abs": [{"id": (ids[k].item() if hasattr(ids[k], "item") else ids[k]),
                                 "difference": float(d[k])} for k in order],
            })
    return out


def _rmsf_window_reliability(tpr_file: str, traj_file: str, index_file: Optional[str], selection: str,
                             res: bool, fit: bool, dt: Optional[float], n_windows: Optional[int],
                             span: Optional[Tuple[float, float]], output_prefix: str,
                             group_input: str) -> Dict[str, Any]:
    """Temporal reliability of the RMSF profile: run gmx rmsf on separate, contiguous, non-overlapping
    windows (same selection, fit, reference, sampling interval) and compare the per-residue profiles.

    Never raises for problems with the reliability analysis itself: the main RMSF result is already
    valid, so skips/failures are reported with a reason code instead. Real data only.
    """
    if n_windows is None:
        return _not_run("not_requested", "Window-based RMSF reliability was not requested "
                        "(pass reliability_windows=N, N >= 2).")
    if span is None:
        return _not_run("end_time_unknown", "The window span is unknown: the whole-trajectory end time is "
                        "not available without reading the trajectory. Pass an end time e (ns) or "
                        "reliability_span=(begin_ns, end_ns).")
    begin, end = span
    try:
        plan = plan_time_windows(begin, end, n_windows)
    except ValueError as exc:
        return _not_run("invalid_window_plan", str(exc), status="inconclusive")
    if dt is not None and (end - begin) / n_windows < 2 * dt:
        return _not_run("insufficient_data", f"Each window ({(end - begin) / n_windows:g} ns) holds fewer than "
                        f"2 frames at the requested sampling interval dt={dt:g} ns.", status="inconclusive")
    labels = [f"{b:g}-{e:g} ns" for b, e in plan]
    runs: List[Dict[str, Any]] = []
    out_paths = [f"{output_prefix}_window{i + 1}.xvg" for i in range(len(plan))]
    for stale in out_paths:
        remove_stale_outputs(stale, stale)
    for i, ((b, e), path) in enumerate(zip(plan, out_paths)):
        w = normalize_time_window(b, e, dt)
        cmd = __import__("execution").gmx_command() + [ "rmsf", "-s", tpr_file, "-f", traj_file, "-o", path]
        if index_file:
            cmd.extend(["-n", index_file])
        cmd.append("-res" if res else "-nores")
        cmd.append("-fit" if fit else "-nofit")
        cmd.extend(time_args(w, "ps"))
        runs.append({"label": labels[i], "begin": b, "end": e, "unit": "ns", "xvg": path,
                     "gmx_command": list(cmd)})
        print(f"[RMSF] Reliability window {i + 1}/{len(plan)} ({labels[i]})...")
        try:
            _execute_rmsf(cmd, group_input)
        except (RuntimeError, FileNotFoundError) as exc:
            return _not_run("window_run_failed", f"gmx rmsf failed for window {labels[i]}.", status="error",
                            error=f"{type(exc).__name__}: {exc}", windows=runs)
    id_col = "Residue" if res else "Atom"
    frames: List[pd.DataFrame] = []
    for r in runs:
        try:
            d = parse_xvg(r["xvg"])
            if d.shape[1] != 2:
                raise ValueError(f"expected 2 columns (index, RMSF), got {d.shape[1]}")
            d.columns = [id_col, "RMSF"]
            require_finite(d, f"RMSF window {r['label']}")
        except (ValueError, FileNotFoundError) as exc:
            return _not_run("window_output_unusable", f"No usable RMSF profile for window {r['label']}: {exc}",
                            status="inconclusive", windows=runs)
        frames.append(d)
    lens = [len(d) for d in frames]
    if len(set(lens)) != 1 or any(not np.array_equal(frames[0][id_col].to_numpy(), d[id_col].to_numpy())
                                  for d in frames[1:]):
        return _not_run("residue_mapping_mismatch", "The windows' RMSF profiles do not cover the same "
                        f"{'residues' if res else 'atoms'} in the same order (profile lengths {lens}); matching "
                        "residues cannot be compared.", status="inconclusive", windows=runs)
    try:
        full = compare_rmsf_windows(frames, labels=labels, windows=plan, window_unit="ns", rmsf_unit="nm")
    except ValueError as exc:
        return _not_run("invalid_profiles", str(exc), status="inconclusive", windows=runs)
    if full["status"] == "insufficient_data":
        return _not_run("insufficient_data", "; ".join(full["warnings"]) or
                        "too few residues/windows for a profile comparison.", status="insufficient_data",
                        windows=runs, warnings=list(full["warnings"]))
    out: Dict[str, Any] = {
        "status": full["status"], "windows": runs,
        "settings": {"selection": selection, "per_residue": res, "fit": fit, "reference": tpr_file,
                     "index_file": index_file, "sampling_interval_ns": dt, "n_windows": len(plan),
                     "span_ns": [begin, end], "window_boundaries": "contiguous, equal length"},
        "summary": dict(full["summary"], n_residues=lens[0], independent_windows=full["independent_windows"],
                        unstable_residue_count=full["unstable_residues"]["count"],
                        unstable_residue_ids=full["unstable_residues"]["ids"]),
        "full": full,
        "per_residue_differences": _per_residue_differences(
            frames[0][id_col].to_numpy(), [d["RMSF"].to_numpy() for d in frames], labels),
        "warnings": list(full["warnings"]) + [
            "GROMACS -b/-e are inclusive: a frame lying exactly on a boundary shared by two adjacent windows "
            "is used in both.",
            "Window agreement shows reproducibility between these windows only, not converged sampling."],
    }
    return out


def run_rmsf(
    tpr_file: str,
    traj_file: str,
    index_file: Optional[str] = None,
    selection: str = "C-alpha",
    res: bool = True,
    b: float = 0.0,
    e: float = -1.0,
    dt: Optional[float] = None,
    output_prefix: str = "rmsf",
    fit: bool = True,
    reliability_windows: Optional[int] = None,
    reliability_span: Optional[Sequence[float]] = None
) -> AnalysisResult:
    """
    Runs gmx rmsf for residue flexibility analysis.
    Scientific Question: Which residues are flexible?
    Default Command: gmx rmsf

    Options (checked against the GROMACS 2022-2026 gmx rmsf manual):
      selection  group for the RMSF calculation (default C-alpha): Backbone, Protein, an individual
                 chain or any custom group. With an index file only groups defined in it are valid.
                 Note -res averages ALL atoms of the group per residue; a chain's C-alpha RMSF needs
                 an index group that already is that intersection.
      res        -res / -nores: per-residue averages (default) or per-atom output.
      fit        -fit / -nofit: least-squares superposition on the .tpr structure before computing
                 RMSF. gmx rmsf fits on the same group it computes RMSF for; it has no separate fit
                 group, no -mw and no PBC option, so the trajectory must already be PBC-corrected.
      reliability_windows  N >= 2 enables the temporal-reliability check: after the main run, gmx rmsf is run
                 on N contiguous, equal-length, non-overlapping windows (same selection, fit, reference and
                 sampling interval) and the per-residue profiles are compared. Off by default because it costs
                 N extra gmx runs. Result: metadata["statistical_reality_check"].
      reliability_span  (begin_ns, end_ns) range split into the windows; defaults to (b, e) and is required
                 when e is not set (the trajectory end is not known without reading it).
      b, e, dt   start / end / sampling interval in ns. gmx rmsf has NO -tu option, its -b/-e/-dt are
                 always in ps, so the values are converted ns -> ps.
    """
    validate_inputs(tpr_file, traj_file, index_file)
    res = require_bool(res, "res")
    fit = require_bool(fit, "fit")
    window = normalize_time_window(b, e, dt)
    selection_ref = resolve_group_ref(selection, index_file, "RMSF selection")
    selection = selection_ref.name
    span: Optional[Tuple[float, float]] = None
    if reliability_windows is not None:
        if isinstance(reliability_windows, bool) or not isinstance(reliability_windows, int) \
                or not (2 <= reliability_windows <= MAX_RELIABILITY_WINDOWS):
            raise ValueError(f"reliability_windows must be an integer between 2 and {MAX_RELIABILITY_WINDOWS} "
                             f"(or None to disable), got {reliability_windows!r}.")
    if reliability_span is not None:
        if reliability_windows is None:
            raise ValueError("reliability_span was given without reliability_windows.")
        try:
            sb, se = (float(v) for v in reliability_span)
        except (TypeError, ValueError):
            raise ValueError("reliability_span must be (begin_ns, end_ns).")
        if not (np.isfinite(sb) and np.isfinite(se)) or sb < window.begin_ns or se <= sb or \
                (window.end_ns is not None and se > window.end_ns):
            raise ValueError(f"reliability_span {tuple(reliability_span)} must satisfy begin < end and lie "
                             f"within the analysis window b={window.begin_ns} ns, e={window.end_ns}.")
        span = (sb, se)
    elif window.end_ns is not None:
        span = (window.begin_ns, window.end_ns)

    output_xvg = f"{output_prefix}.xvg"
    output_csv = f"{output_prefix}.csv"
    remove_stale_outputs(output_xvg, output_csv)

    cmd = __import__("execution").gmx_command() + [ "rmsf", "-s", tpr_file, "-f", traj_file, "-o", output_xvg]
    if index_file:
        cmd.extend(["-n", index_file])
    cmd.append("-res" if res else "-nores")
    cmd.append("-fit" if fit else "-nofit")
    cmd.extend(time_args(window, "ps"))

    group_input = f"{selection_ref.stdin_token}\n"   # group number with an index file, else the name

    warnings = [
        "gmx rmsf has no PBC handling: the trajectory must already have molecules whole and no "
        "periodic jumps, otherwise RMSF of affected residues is inflated by boundary artifacts.",
    ]
    if not fit:
        warnings.append("fit=False: no superposition; the trajectory must already match the reference "
                        "structure, otherwise global motion is counted as fluctuation.")

    print(f"[RMSF] Executing gmx rmsf for selection '{selection}'...")
    _execute_rmsf(cmd, group_input)

    df = parse_xvg(output_xvg)
    if df.shape[1] != 2:
        raise ValueError(f"Unexpected gmx rmsf output: expected 2 columns (index, RMSF), got {df.shape[1]}.")
    df.columns = ["Residue" if res else "Atom", "RMSF"]
    require_finite(df, "RMSF")
    df.to_csv(output_csv, index=False)

    val_series = df["RMSF"].values
    summary = compute_basic_summary(val_series)
    x_unit = "Residue Index" if res else "Atom Index"

    reality = _rmsf_window_reliability(tpr_file, traj_file, index_file, selection, res, fit, window.step_ns,
                                       reliability_windows, span, output_prefix, group_input)
    if reality["status"] in ("not_run", "error", "inconclusive", "insufficient_data") and "reason" in reality:
        if reality.get("reason_code") != "not_requested":
            warnings.append(f"Statistical reality check {reality['status'].replace('_', ' ')}: {reality['reason']}")
    elif reality["status"] != "not_run":
        warnings.extend(f"Statistical check: {w}" for w in reality["warnings"])

    metadata = {
        "scientific_question": "Which residues are flexible?",
        "analysis_type": "RMSF",
        "command": "gmx rmsf",
        "status": "SUCCESS",
        "selection": selection,
        "per_residue": res,
        "x_unit": x_unit,
        "y_unit": "nm",
        "tpr_file": tpr_file,
        "traj_file": traj_file,
        "gmx_version": get_gromacs_version_info()["version_string"],
        "xmgrace_cmd": get_xmgrace_command(output_xvg),
        # --- TUI integration: selected parameters, units, provenance ---
        "index_file": index_file,
        "fit": fit,
        "parameters": {
            "selection": selection, "per_residue": res, "fit": fit, "fit_group": selection if fit else None,
            "reference": tpr_file, "time_window": window.as_dict(),
        },
        "units": {"x": x_unit, "rmsf": "nm", "time_window_input": "ns",
                  "time_passed_to_gmx": "ps (converted from ns; gmx rmsf has no -tu)"},
        "gmx_command": list(cmd),
        "gmx_stdin_groups": [selection],
        "gmx_stdin_input": [selection_ref.stdin_token],
        "group_resolution": {"selection": selection_ref.as_dict()},
        "warnings": warnings,
        "statistical_reality_check": reality,
    }

    return AnalysisResult(
        analysis_name="RMSF",
        output_xvg=output_xvg,
        output_csv=output_csv,
        data=df,
        metadata=metadata,
        statistics=summary
    )
