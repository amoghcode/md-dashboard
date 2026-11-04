"""Integration tests: statistical_reality_check inside run_rmsd.

MOCKED: `gmx rms` is replaced by a fake that writes a canned .xvg; the RMSD values are synthetic
(AR(1) noise with a decaying transient). These tests verify wiring (actual values/dt/units passed,
results attached, failures handled). They are NOT validation on a real GROMACS RMSD run.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis_backend import rmsd as rmsd_mod
from analysis_backend.rmsd import run_rmsd
from analysis_backend.statistics import statistical_reality_check as real_check

V = {"version_string": "GROMACS version:    2026.2", "major_year": 2026, "available": True}


def synthetic_rmsd(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + e[i]
    t = np.arange(n)
    return 0.25 + 0.004 * x + 0.15 * np.exp(-t / 120.0)   # nm


def xvg_text(times, values):
    head = '@ xaxis  label "Time (ns)"\n@ yaxis  label "RMSD (nm)"\n'
    return head + "".join(f"{t:.6f} {v:.8f}\n" for t, v in zip(times, values))


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.tpr = self._f("topol.tpr")
        self.traj = self._f("traj.xtc")
        self.prefix = os.path.join(self.d, "rmsd")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _f(self, name):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            fh.write("x")
        return p

    def run_with(self, times, values, check=None):
        def fake_run(cmd, log_prefix="gmx", stdin_input=None):
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write(xvg_text(times, values))
            return ""
        patches = [patch("analysis_backend.rmsd.run_gromacs_command", fake_run),
                   patch("analysis_backend.rmsd.get_gromacs_version_info", return_value=V)]
        if check is not None:
            patches.append(patch("analysis_backend.rmsd.statistical_reality_check", check))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return run_rmsd(self.tpr, self.traj, output_prefix=self.prefix)


class TestSuccessfulIntegration(Base):
    def test_actual_values_dt_and_units_are_passed(self):
        values = synthetic_rmsd()
        times = np.arange(len(values)) * 0.1          # 0.1 ns spacing
        seen = {}

        def spy(series, **kw):
            seen["series"], seen["kw"] = np.array(series), kw
            return real_check(series, **kw)
        res = self.run_with(times, values, check=spy)
        self.assertEqual(seen["series"].shape, values.shape)
        np.testing.assert_allclose(seen["series"], res.data.iloc[:, 1].to_numpy())  # the values in the CSV
        np.testing.assert_allclose(seen["series"], values, atol=1e-7)                # the values gmx wrote
        self.assertAlmostEqual(seen["kw"]["dt"], 0.1)
        self.assertEqual(seen["kw"]["time_unit"], "ns")
        self.assertEqual(seen["kw"]["value_unit"], "nm")

    def test_results_attached_to_metadata(self):
        values = synthetic_rmsd()
        res = self.run_with(np.arange(len(values)) * 0.1, values)
        rc = res.metadata["statistical_reality_check"]
        self.assertIn(rc["status"], ("no_issues_detected", "inconclusive"))
        s = rc["summary"]
        self.assertEqual(s["candidate_equilibration"]["time_unit"], "ns")
        self.assertGreater(s["candidate_equilibration"]["t0_frame"], 0)       # transient was detected
        self.assertAlmostEqual(s["candidate_equilibration"]["t0_time"],
                               s["candidate_equilibration"]["t0_frame"] * 0.1)
        self.assertGreater(s["ess"], 0)
        self.assertLess(s["ess"], s["analysis_segment"]["n_frames"])           # correlated -> ESS < N
        self.assertGreater(s["sem_corrected"], s["sem_naive"])
        self.assertIn(s["block_convergence"]["status"], ("converged", "not_converged", "insufficient_data"))
        self.assertEqual(s["value_unit"], "nm")
        for key in ("equilibration", "autocorrelation", "effective_sample_size", "block_averaging",
                    "cross_checks", "overall", "limitations"):
            self.assertIn(key, rc["full"])                                     # complete diagnostics kept
        self.assertEqual(rc["frame_spacing"], {"dt": rc["frame_spacing"]["dt"], "unit": "ns", "n_frames": 3000})
        json.dumps(rc)

    def test_existing_outputs_preserved(self):
        values = synthetic_rmsd(500)
        res = self.run_with(np.arange(500) * 0.1, values)
        m = res.metadata
        for key in ("analysis_type", "command", "fit_group", "calc_group", "ref_structure", "mass_weighted",
                    "fit_mode", "pbc", "time_unit", "parameters", "units", "gmx_command", "time_coverage",
                    "warnings"):
            self.assertIn(key, m)
        self.assertEqual(set(res.statistics), {"mean", "std", "min", "max", "median"})
        self.assertAlmostEqual(res.statistics["mean"], float(np.mean(values)), places=6)
        self.assertTrue(os.path.exists(res.output_csv) and os.path.exists(res.output_xvg))
        self.assertEqual(res.data.shape, (500, 2))

    def test_check_warnings_surface_in_metadata_warnings(self):
        res = self.run_with(np.arange(30) * 0.1, np.linspace(0.1, 0.2, 30) + 0.001 * np.arange(30) % 3)
        rc = res.metadata["statistical_reality_check"]
        self.assertNotEqual(rc["status"], "no_issues_detected")                # 30 frames cannot be reliable
        self.assertTrue(any(w.startswith("Statistical check:") for w in res.metadata["warnings"]))

    def test_short_trajectory_reported_honestly(self):
        res = self.run_with([0.0, 0.5, 1.0], [0.0, 0.12, 0.15])
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "insufficient_data")
        self.assertIsNone(rc["summary"]["ess"])


class TestTimestamps(Base):
    def check(self, offset):
        values = synthetic_rmsd()
        times = offset + np.arange(len(values)) * 0.1
        res = self.run_with(times, values)
        eq = res.metadata["statistical_reality_check"]["summary"]["candidate_equilibration"]
        self.assertGreater(eq["t0_frame"], 0)
        self.assertAlmostEqual(eq["t0_time"], offset + eq["t0_frame"] * 0.1, places=4)   # actual timestamp
        self.assertAlmostEqual(eq["t0_elapsed"], eq["t0_frame"] * 0.1, places=4)
        self.assertEqual(eq["time_origin"], "supplied_times")
        seg = res.metadata["statistical_reality_check"]["summary"]["analysis_segment"]
        self.assertAlmostEqual(seg["start_time"], eq["t0_time"], places=4)
        full_inputs = res.metadata["statistical_reality_check"]["full"]["inputs"]
        self.assertAlmostEqual(full_inputs["start_time"], offset, places=4)
        return eq

    def test_run_starting_at_zero_ns(self):
        eq = self.check(0.0)
        self.assertAlmostEqual(eq["t0_time"], eq["t0_elapsed"], places=4)

    def test_run_starting_at_200_ns(self):
        eq = self.check(200.0)
        self.assertGreaterEqual(eq["t0_time"], 200.0)
        self.assertNotAlmostEqual(eq["t0_time"], eq["t0_elapsed"], places=1)

    def test_actual_time_column_is_passed_to_check(self):
        seen = {}

        def spy(series, **kw):
            seen.update(kw)
            return real_check(series, **kw)
        values = synthetic_rmsd(500)
        times = 200.0 + np.arange(500) * 0.1
        self.run_with(times, values, check=spy)
        np.testing.assert_allclose(seen["times"], times, atol=1e-5)


class TestFailureHandling(Base):
    def test_check_exception_is_recorded_rmsd_still_returned(self):
        def boom(*a, **k):
            raise RuntimeError("numerical failure")
        values = synthetic_rmsd(400)
        res = self.run_with(np.arange(400) * 0.1, values, check=boom)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "error")
        self.assertIn("numerical failure", rc["error"])
        self.assertNotIn("summary", rc)                                        # no invented numbers
        self.assertTrue(any("reality check error" in w for w in res.metadata["warnings"]))
        self.assertEqual(res.data.shape, (400, 2))                             # RMSD result intact
        self.assertTrue(os.path.exists(res.output_csv))

    def test_non_uniform_frame_spacing_skips_check(self):
        calls = []
        times = np.concatenate([np.arange(0, 20, 0.1), np.arange(20, 60, 1.0)])
        values = synthetic_rmsd(len(times))
        res = self.run_with(times, values, check=lambda *a, **k: calls.append(1))
        self.assertEqual(calls, [])
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "not_run")
        self.assertIn("not equally spaced", rc["reason"])
        self.assertTrue(any("not run" in w for w in res.metadata["warnings"]))

    def test_gromacs_failure_means_no_check_and_no_result(self):
        called = []
        with patch("analysis_backend.rmsd.run_gromacs_command", side_effect=RuntimeError("gmx failed")), \
             patch("analysis_backend.rmsd.get_gromacs_version_info", return_value=V), \
             patch("analysis_backend.rmsd.statistical_reality_check", lambda *a, **k: called.append(1)):
            with self.assertRaises(RuntimeError):
                run_rmsd(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertEqual(called, [])                                           # never run on missing data
        self.assertFalse(os.path.exists(self.prefix + ".csv"))

    def test_nan_output_still_fails_before_any_check(self):
        called = []

        def fake_run(cmd, log_prefix="gmx", stdin_input=None):
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write('@ xaxis  label "Time (ns)"\n0.0 nan\n0.1 0.2\n')
            return ""
        with patch("analysis_backend.rmsd.run_gromacs_command", fake_run), \
             patch("analysis_backend.rmsd.get_gromacs_version_info", return_value=V), \
             patch("analysis_backend.rmsd.statistical_reality_check", lambda *a, **k: called.append(1)):
            with self.assertRaises(ValueError):
                run_rmsd(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
