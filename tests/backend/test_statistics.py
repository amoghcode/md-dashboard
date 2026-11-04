"""Tests for analysis_backend.statistics (Statistical Reality Check).

WHAT THESE TESTS ARE: synthetic-data tests. Series are generated with fixed seeds from processes with
known analytic answers (white noise, AR(1) with tau_int = (1+phi)/(2(1-phi)), exponential transient,
linear drift), so the estimators are checked against theory. The pymbar test injects a FAKE pymbar
module and only checks plumbing; the real pymbar package was not available when these were written.

WHAT THEY ARE NOT: validation on a real GROMACS simulation. No MD trajectory was analysed.
"""
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis_backend import statistics as S
from analysis_backend.statistics import compute_basic_summary


def ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def tau_theory(phi):
    return 0.5 * (1 + phi) / (1 - phi)


class TestExistingBehaviourPreserved(unittest.TestCase):
    def test_basic_summary_unchanged(self):
        s = compute_basic_summary(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(s, {"mean": 2.5, "std": float(np.std([1, 2, 3, 4])), "min": 1.0, "max": 4.0, "median": 2.5})
        self.assertEqual(compute_basic_summary(np.array([])), {})


class TestAutocorrelation(unittest.TestCase):
    def test_white_noise(self):
        r = S.integrated_autocorrelation_time(ar1(20000, 0.0, 11), dt=0.1, time_unit="ns")
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["tau_int_frames"], 0.5, delta=0.1)
        self.assertAlmostEqual(r["statistical_inefficiency"], 1.0, delta=0.2)

    def test_ar1_matches_theory_and_converts_time(self):
        for phi, seed in [(0.5, 21), (0.9, 22)]:
            r = S.integrated_autocorrelation_time(ar1(60000, phi, seed), dt=0.2, time_unit="ns")
            th = tau_theory(phi)
            self.assertEqual(r["status"], "ok")
            self.assertLess(abs(r["tau_int_frames"] - th) / th, 0.2, (phi, r["tau_int_frames"], th))
            self.assertAlmostEqual(r["tau_int_time"], r["tau_int_frames"] * 0.2)
            self.assertEqual(r["time_unit"], "ns")
            # theory is within a few reported standard errors
            self.assertLess(abs(r["tau_int_frames"] - th), 4 * r["tau_int_stderr_frames"])

    def test_acf_function(self):
        rho = S.autocorrelation_function(ar1(60000, 0.8, 5), max_lag=3)
        self.assertEqual(rho[0], 1.0)
        for k in (1, 2, 3):
            self.assertAlmostEqual(rho[k], 0.8 ** k, delta=0.03)

    def test_include_curve(self):
        r = S.integrated_autocorrelation_time(ar1(5000, 0.5, 1), include_curve=True)
        self.assertEqual(r["acf"]["values"][0], 1.0)
        self.assertEqual(len(r["acf"]["lags_frames"]), len(r["acf"]["values"]))

    def test_series_too_short_for_its_correlation_time_is_inconclusive(self):
        # true tau_int ~ 100 frames; with N=500 the estimator can badly underestimate it
        # (seed 3 gives ~10), so the status must not be 'ok'.
        for seed in (3, 4, 5):
            r = S.integrated_autocorrelation_time(ar1(500, 0.995, seed))
            self.assertEqual(r["status"], "inconclusive", seed)
            self.assertTrue(r["warnings"])

    def test_insufficient_and_degenerate_inputs(self):
        r = S.integrated_autocorrelation_time(np.arange(10.0))
        self.assertEqual(r["status"], "insufficient_data")
        self.assertIsNone(r["tau_int_frames"])
        r = S.integrated_autocorrelation_time(np.ones(100))
        self.assertEqual(r["status"], "inconclusive")
        self.assertIsNone(r["tau_int_frames"])
        with self.assertRaises(ValueError):
            S.autocorrelation_function(np.ones(10))
        with self.assertRaises(ValueError):
            S.integrated_autocorrelation_time(np.array([1.0, np.nan, 2.0] * 20))
        with self.assertRaises(ValueError):
            S.integrated_autocorrelation_time(np.ones((5, 5)))


class TestESS(unittest.TestCase):
    def test_white_noise_ess_equals_n(self):
        x = ar1(20000, 0.0, 31)
        r = S.effective_sample_size(x, value_unit="nm")
        self.assertAlmostEqual(r["ess"] / 20000, 1.0, delta=0.15)
        self.assertAlmostEqual(r["sem_corrected"] / r["sem_naive"], 1.0, delta=0.1)
        self.assertEqual(r["value_unit"], "nm")

    def test_ar1_ess_and_sem_inflation(self):
        phi, n = 0.9, 60000
        r = S.effective_sample_size(ar1(n, phi, 32), dt=1.0, time_unit="ps", value_unit="nm")
        g = (1 + phi) / (1 - phi)
        self.assertLess(abs(r["ess"] - n / g) / (n / g), 0.2)
        self.assertLess(abs(r["sem_inflation_factor"] - np.sqrt(g)) / np.sqrt(g), 0.1)
        self.assertGreater(r["sem_corrected"], 3 * r["sem_naive"])
        self.assertLessEqual(r["ess"], n)

    def test_low_ess_flagged(self):
        r = S.effective_sample_size(ar1(2000, 0.99, 7))
        self.assertNotEqual(r["status"], "ok")
        self.assertTrue(r["warnings"])

    def test_insufficient(self):
        r = S.effective_sample_size(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(r["status"], "insufficient_data")
        self.assertIsNone(r["ess"])
        self.assertIsNone(r["sem_corrected"])


class TestEquilibration(unittest.TestCase):
    def transient(self, n=5000, tau=100.0, amp=5.0, seed=4):
        t = np.arange(n)
        return ar1(n, 0.5, seed) * 0.2 + amp * np.exp(-t / tau)

    def test_detects_exponential_transient(self):
        r = S.detect_equilibration(self.transient(), dt=0.1, time_unit="ns", method="native")
        self.assertEqual(r["method"], "native")
        self.assertEqual(r["status"], "candidate")
        self.assertGreater(r["t0_frame"], 100)     # must discard a substantial part of the transient
        self.assertLess(r["t0_frame"], 1500)
        self.assertAlmostEqual(r["t0_time"], r["t0_frame"] * 0.1)
        self.assertEqual(r["n_discarded"] + r["n_remaining"], 5000)
        self.assertIn("cannot prove", r["interpretation"])

    def test_stationary_series_discards_little(self):
        r = S.detect_equilibration(ar1(5000, 0.5, 8), method="native")
        self.assertLess(r["fraction_discarded"], 0.5)

    def test_persistent_drift_is_not_called_equilibrated(self):
        x = np.linspace(0, 10, 3000) + np.random.default_rng(5).normal(0, 0.3, 3000)
        r = S.detect_equilibration(x, method="native")
        self.assertEqual(r["status"], "inconclusive")
        self.assertTrue(r["warnings"])

    def test_insufficient_and_constant(self):
        self.assertEqual(S.detect_equilibration(np.arange(10.0))["status"], "insufficient_data")
        r = S.detect_equilibration(np.ones(200))
        self.assertEqual(r["status"], "inconclusive")

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            S.detect_equilibration(np.arange(100.0), method="magic")
        with self.assertRaises(ValueError):
            S.detect_equilibration(np.arange(100.0), max_discard_fraction=1.5)

    def test_auto_falls_back_to_native_with_warning_when_pymbar_missing(self):
        with patch.dict(sys.modules, {"pymbar": None}):  # makes `import pymbar` raise ImportError
            r = S.detect_equilibration(self.transient(), method="auto")
            self.assertEqual(r["method"], "native")
            self.assertTrue(any("pymbar is not installed" in w for w in r["warnings"]))
            with self.assertRaises(ImportError):
                S.detect_equilibration(self.transient(), method="pymbar")

    def test_pymbar_path_plumbing_with_FAKE_module(self):
        """MOCKED: fake pymbar; verifies arguments and use of its return values only."""
        calls = {}
        fake = types.ModuleType("pymbar")
        fake_ts = types.ModuleType("pymbar.timeseries")

        def detect_equilibration(a, fast=True, nskip=1):
            calls.update(n=len(a), fast=fast, nskip=nskip)
            return 40, 7.0, 123.0
        fake_ts.detect_equilibration = detect_equilibration
        fake.timeseries = fake_ts
        with patch.dict(sys.modules, {"pymbar": fake, "pymbar.timeseries": fake_ts}):
            r = S.detect_equilibration(ar1(2000, 0.3, 1), dt=2.0, method="pymbar", nskip=10)
        self.assertEqual(r["method"], "pymbar")
        self.assertEqual((r["t0_frame"], r["statistical_inefficiency_at_t0"], r["ess_at_t0"]), (40, 7.0, 123.0))
        self.assertEqual(r["t0_time"], 80.0)
        self.assertEqual(calls, {"n": 2000, "fast": True, "nskip": 10})


class TestBlockAveraging(unittest.TestCase):
    def test_correlated_series_converges_to_theory(self):
        phi, n = 0.9, 65536
        r = S.block_averaging(ar1(n, phi, 41), dt=1.0, time_unit="ps", value_unit="nm")
        c = r["convergence"]
        self.assertEqual(c["status"], "converged")
        g = (1 + phi) / (1 - phi)
        self.assertLess(abs(c["implied_ess"] - n / g) / (n / g), 0.4)
        sems = [b["sem"] for b in r["blocks"]]
        self.assertGreater(sems[-3], 2.5 * sems[0])        # SEM grows with block size for correlated data
        self.assertEqual([b["block_size_frames"] for b in r["blocks"]][:4], [1, 2, 4, 8])
        for b in r["blocks"]:
            self.assertEqual(b["n_blocks"], n // b["block_size_frames"])
            self.assertEqual(b["n_discarded_frames"], n - b["n_blocks"] * b["block_size_frames"])
            self.assertGreater(b["sem_uncertainty"], 0)

    def test_white_noise_sem_flat(self):
        r = S.block_averaging(ar1(32768, 0.0, 43))
        for b in r["blocks"][:6]:
            self.assertAlmostEqual(b["sem"] / r["sem_naive"], 1.0, delta=0.25)

    def test_slow_series_not_converged(self):
        r = S.block_averaging(ar1(2000, 0.999, 2))
        self.assertEqual(r["convergence"]["status"], "not_converged")
        self.assertIsNone(r["convergence"]["sem_plateau"])
        self.assertTrue(r["warnings"])

    def test_custom_block_sizes_and_errors(self):
        r = S.block_averaging(ar1(1000, 0.5, 1), block_sizes=[10, 50, 10])
        self.assertEqual([b["block_size_frames"] for b in r["blocks"]], [10, 50])
        self.assertEqual(r["convergence"]["status"], "insufficient_data")
        with self.assertRaises(ValueError):
            S.block_averaging(np.arange(100.0), block_sizes=[0])
        with self.assertRaises(ValueError):
            S.block_averaging(np.arange(100.0), block_sizes=[80])

    def test_insufficient_and_constant(self):
        self.assertEqual(S.block_averaging(np.arange(10.0))["convergence"]["status"], "insufficient_data")
        self.assertEqual(S.block_averaging(np.ones(100))["convergence"]["status"], "inconclusive")


class TestRealityCheck(unittest.TestCase):
    def test_well_sampled_correlated_series(self):
        x = ar1(65536, 0.9, 52)  # note: ~10% of seeds give a (honest) 'not_converged' block plateau
        r = S.statistical_reality_check(x, dt=0.01, time_unit="ns", value_unit="nm")
        self.assertEqual(r["overall"]["status"], "no_issues_detected")
        self.assertTrue(r["cross_checks"]["agreement"])
        self.assertEqual(r["inputs"]["value_unit"], "nm")
        self.assertEqual(r["autocorrelation"]["time_unit"], "ns")
        self.assertGreater(len(r["limitations"]), 3)
        self.assertIn("not proof", r["overall"]["meaning"])
        json.dumps(r)  # fully JSON-serialisable for TUI/metadata use

    def test_transient_series_uses_post_candidate_segment(self):
        n = 5000
        x = ar1(n, 0.5, 4) * 0.2 + 5 * np.exp(-np.arange(n) / 100.0)
        r = S.statistical_reality_check(x, dt=1.0, time_unit="ps", equilibration_method="native")
        self.assertEqual(r["equilibration"]["status"], "candidate")
        seg = r["analysis_segment"]
        self.assertEqual(seg["start_frame"], r["equilibration"]["t0_frame"])
        self.assertEqual(seg["n_frames"], n - seg["start_frame"])
        self.assertGreater(seg["start_frame"], 0)

    def test_short_slow_series_is_inconclusive_not_ok(self):
        r = S.statistical_reality_check(ar1(2000, 0.999, 2), dt=1.0)
        self.assertEqual(r["overall"]["status"], "inconclusive")
        self.assertTrue(r["overall"]["reasons"])
        self.assertTrue(r["warnings"])

    def test_drifting_series_not_reported_as_fine(self):
        x = np.linspace(0, 10, 3000) + np.random.default_rng(5).normal(0, 0.3, 3000)
        r = S.statistical_reality_check(x, dt=1.0)
        self.assertEqual(r["overall"]["status"], "inconclusive")
        self.assertEqual(r["analysis_segment"]["basis"], "full series")

    def test_insufficient_data(self):
        r = S.statistical_reality_check(np.arange(10.0), dt=1.0)
        self.assertEqual(r["overall"]["status"], "insufficient_data")
        for sec in ("autocorrelation", "effective_sample_size"):
            self.assertIsNone(r[sec].get("tau_int_frames"))
        self.assertEqual(S.statistical_reality_check(np.array([]))["overall"]["status"], "insufficient_data")

    def test_constant_series_inconclusive(self):
        self.assertEqual(S.statistical_reality_check(np.ones(300))["overall"]["status"], "inconclusive")

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            S.statistical_reality_check(np.arange(100.0), dt=-1)
        with self.assertRaises(ValueError):
            S.statistical_reality_check([1.0, float("nan")] * 50)

    def test_accepts_pandas_series(self):
        r = S.statistical_reality_check(pd.Series(ar1(3000, 0.5, 2)), dt=1.0)
        self.assertEqual(r["inputs"]["n_frames"], 3000)

    def test_infer_timestep(self):
        self.assertTrue(S.infer_timestep(np.arange(0, 100, 0.5))["uniform"])
        self.assertAlmostEqual(S.infer_timestep(np.arange(0, 100, 0.5))["dt"], 0.5)
        bad = S.infer_timestep([0, 1, 2, 4, 5])
        self.assertFalse(bad["uniform"])
        self.assertFalse(S.infer_timestep([0.0])["uniform"])
        self.assertFalse(S.infer_timestep([3, 2, 1])["uniform"])


class TestEquilibrationTimestamps(unittest.TestCase):
    """t0_frame is an index into the series; t0_time must be the actual simulation timestamp."""

    def series(self):
        n = 5000
        return ar1(n, 0.5, 4) * 0.2 + 5 * np.exp(-np.arange(n) / 100.0)

    def test_trajectory_starting_at_zero(self):
        x, dt = self.series(), 0.1
        r = S.detect_equilibration(x, dt=dt, time_unit="ns", method="native", times=np.arange(len(x)) * dt)
        self.assertGreater(r["t0_frame"], 0)
        self.assertAlmostEqual(r["t0_time"], r["t0_frame"] * dt)
        self.assertAlmostEqual(r["t0_elapsed"], r["t0_frame"] * dt)
        self.assertEqual(r["time_origin"], "supplied_times")
        self.assertEqual(r["series_start_time"], 0.0)

    def test_trajectory_starting_at_200_ns(self):
        x, dt, t0ns = self.series(), 0.1, 200.0
        base = S.detect_equilibration(x, dt=dt, time_unit="ns", method="native")
        for kw in (dict(times=t0ns + np.arange(len(x)) * dt), dict(t_start=t0ns)):
            r = S.detect_equilibration(x, dt=dt, time_unit="ns", method="native", **kw)
            self.assertEqual(r["t0_frame"], base["t0_frame"])                 # frame index unchanged
            self.assertAlmostEqual(r["t0_time"], t0ns + r["t0_frame"] * dt)   # timestamp is offset
            self.assertAlmostEqual(r["t0_elapsed"], r["t0_frame"] * dt)       # elapsed is not
            self.assertGreaterEqual(r["t0_time"], 200.0)
            self.assertEqual(r["series_start_time"], 200.0)
        self.assertEqual(S.detect_equilibration(x, dt=dt, method="native", t_start=t0ns)["time_origin"],
                         "supplied_t_start")

    def test_default_without_offset_is_labelled_assumed_zero(self):
        x = self.series()
        r = S.detect_equilibration(x, dt=0.1, method="native")
        self.assertEqual(r["time_origin"], "assumed_zero")
        self.assertAlmostEqual(r["t0_time"], r["t0_frame"] * 0.1)

    def test_explicit_timestamps_win_over_dt_arithmetic(self):
        x, dt = self.series(), 0.1
        times = 200.0 + np.arange(len(x)) * dt
        times[1000:] += 7.0                                                  # a gap: dt arithmetic would be wrong
        r = S.detect_equilibration(x, dt=dt, method="native", times=times)
        self.assertEqual(r["t0_time"], float(times[r["t0_frame"]]))

    def test_invalid_times_raise(self):
        x = self.series()
        with self.assertRaises(ValueError):
            S.detect_equilibration(x, dt=0.1, times=np.arange(10.0))
        bad = np.arange(len(x)) * 0.1
        bad[5] = bad[4]
        with self.assertRaises(ValueError):
            S.detect_equilibration(x, dt=0.1, times=bad)

    def test_insufficient_data_still_reports_start_time_and_no_t0(self):
        r = S.detect_equilibration(np.arange(10.0), dt=1.0, t_start=200.0)
        self.assertEqual(r["status"], "insufficient_data")
        self.assertIsNone(r["t0_time"])
        self.assertEqual(r["series_start_time"], 200.0)

    def test_orchestrator_reports_timestamps_consistently(self):
        x, dt = self.series(), 0.1
        for t_off in (0.0, 200.0):
            times = t_off + np.arange(len(x)) * dt
            r = S.statistical_reality_check(x, dt=dt, time_unit="ns", equilibration_method="native", times=times)
            eq, seg = r["equilibration"], r["analysis_segment"]
            self.assertAlmostEqual(eq["t0_time"], t_off + eq["t0_frame"] * dt)
            self.assertAlmostEqual(seg["start_time"], eq["t0_time"])          # segment starts at the same stamp
            self.assertEqual(seg["start_frame"], eq["t0_frame"])
            self.assertAlmostEqual(r["inputs"]["start_time"], t_off)
            self.assertAlmostEqual(r["inputs"]["end_time"], t_off + (len(x) - 1) * dt)
            self.assertAlmostEqual(r["inputs"]["duration"], (len(x) - 1) * dt)  # duration is elapsed, offset-free
        r = S.statistical_reality_check(x, dt=dt, equilibration_method="native", t_start=200.0)
        self.assertAlmostEqual(r["analysis_segment"]["start_time"], 200.0 + r["equilibration"]["t0_frame"] * dt)


class TestRMSFWindows(unittest.TestCase):
    def profile(self, seed, n=120, noise=0.02):
        rng = np.random.default_rng(seed)
        x = np.arange(n)
        base = 0.1 + 0.3 * np.exp(-((x - 30) / 5.0) ** 2) + 0.5 * np.exp(-((x - 90) / 6.0) ** 2)  # flexible loops
        return base + rng.normal(0, noise, n).clip(-0.05, 0.05)

    def test_reproducible_profiles_are_consistent(self):
        ps = [self.profile(1), self.profile(2), self.profile(3)]
        r = S.compare_rmsf_windows(ps, labels=["0-100", "100-200", "200-300"],
                                   windows=[(0, 100), (100, 200), (200, 300)])
        self.assertEqual(r["status"], "consistent")
        self.assertEqual(len(r["pairwise"]), 3)
        self.assertGreater(r["summary"]["min_pearson"], 0.95)
        self.assertGreater(r["summary"]["mean_top_flexible_jaccard"], 0.5)
        self.assertTrue(r["independent_windows"])
        self.assertEqual(r["rmsf_unit"], "nm")
        self.assertEqual(len(r["per_residue"]["mean"]), 120)
        json.dumps(r)

    def test_unrelated_profiles_are_inconsistent(self):
        rng = np.random.default_rng(0)
        r = S.compare_rmsf_windows([rng.uniform(0.05, 0.6, 150), rng.uniform(0.05, 0.6, 150)])
        self.assertEqual(r["status"], "inconsistent")
        self.assertLess(r["summary"]["min_pearson"], 0.3)

    def test_changed_flexible_region_is_partial_or_worse_and_residues_flagged(self):
        a = self.profile(1, noise=0.003)
        b = self.profile(2, noise=0.003)
        b[85:100] += 0.6 * np.exp(-((np.arange(15) - 7) / 3.0) ** 2)   # a loop that opens only in window 2
        r = S.compare_rmsf_windows([a, b], flag_cv=0.4)
        self.assertNotEqual(r["status"], "consistent")
        self.assertGreater(r["unstable_residues"]["count"], 0)
        top = r["unstable_residues"]["ids"][0]
        self.assertTrue(86 <= top <= 101, top)  # ids are 1-based

    def test_single_profile_is_insufficient_never_split_into_blocks(self):
        r = S.compare_rmsf_windows([self.profile(1)])
        self.assertEqual(r["status"], "insufficient_data")
        self.assertNotIn("pairwise", r)
        self.assertTrue(any("cannot be treated as time blocks" in w for w in r["warnings"]))
        self.assertEqual(S.compare_rmsf_windows([])["status"], "insufficient_data")

    def test_mismatched_inputs_raise(self):
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([self.profile(1, n=100), self.profile(2, n=90)])
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([np.full(50, np.nan), np.ones(50)])
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([-np.ones(50), np.ones(50)])
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([self.profile(1), self.profile(2)], labels=["only one"])
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([self.profile(1), self.profile(2)], windows=[(0, 1)])
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([self.profile(1), self.profile(2)], windows=[(0, 1), (5, 5)])

    def test_overlapping_or_unequal_windows_warn_and_downgrade(self):
        ps = [self.profile(1), self.profile(2)]
        r = S.compare_rmsf_windows(ps, windows=[(0, 100), (50, 150)])
        self.assertFalse(r["independent_windows"])
        self.assertEqual(r["status"], "inconclusive")
        r = S.compare_rmsf_windows(ps, windows=[(0, 100), (100, 300)])
        self.assertTrue(any("duration" in w for w in r["warnings"]))

    def test_constant_profile_is_inconclusive(self):
        r = S.compare_rmsf_windows([np.full(50, 0.2), self.profile(1, n=50)])
        self.assertEqual(r["status"], "inconclusive")

    def test_too_few_residues_insufficient(self):
        r = S.compare_rmsf_windows([np.arange(1, 6) / 10.0, np.arange(1, 6) / 10.0])
        self.assertEqual(r["status"], "insufficient_data")

    def test_dataframes_from_run_rmsf_and_identifier_checks(self):
        ids = np.arange(1, 61)
        d1 = pd.DataFrame({"Residue": ids, "RMSF": self.profile(1, n=60)})
        d2 = pd.DataFrame({"Residue": ids, "RMSF": self.profile(2, n=60)})
        r = S.compare_rmsf_windows([d1, d2])
        self.assertEqual(r["per_residue"]["ids"][:3], [1, 2, 3])
        d3 = d2.copy()
        d3["Residue"] = ids + 1
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([d1, d3])
        with self.assertRaises(ValueError):
            S.compare_rmsf_windows([d1, pd.DataFrame({"X": ids})])

    def test_plan_time_windows(self):
        w = S.plan_time_windows(100.0, 500.0, 4)
        self.assertEqual(w, [(100.0, 200.0), (200.0, 300.0), (300.0, 400.0), (400.0, 500.0)])
        for bad in [(0, 10, 1), (10, 5, 2), (-1, 5, 2)]:
            with self.assertRaises(ValueError):
                S.plan_time_windows(*bad)


if __name__ == "__main__":
    unittest.main()
