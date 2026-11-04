"""Integration tests: statistical_reality_check inside run_gyrate.

MOCKED: `gmx gyrate` is replaced by a fake that writes a canned .xvg (times in ps, as gmx writes them
without -tu). Rg values are synthetic (correlated noise + decaying transient). These tests verify wiring
only; they are NOT validation on a real GROMACS Rg run.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis_backend.radius_of_gyration import run_gyrate, _total_rg_column
from analysis_backend.statistics import statistical_reality_check as real_check

V = {"version_string": "GROMACS version:    2026.2", "major_year": 2026, "available": True}
LEGENDS = '@ s0 legend "Rg"\n@ s1 legend "RgX"\n@ s2 legend "RgY"\n@ s3 legend "RgZ"\n'


def synthetic_rg(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + e[i]
    return 2.60 + 0.01 * x + 0.20 * np.exp(-np.arange(n) / 120.0)   # nm


def xvg(times_ps, rg, legends=LEGENDS, x_label=None):
    head = (f'@ xaxis  label "{x_label}"\n' if x_label else "") + legends
    # components deliberately different from total Rg so a wrong column would be caught
    return head + "".join(f"{t:.4f} {r:.8f} {r * 0.5:.8f} {r * 0.6:.8f} {r * 0.7:.8f}\n"
                          for t, r in zip(times_ps, rg))


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.tpr = self._f("topol.tpr")
        self.traj = self._f("traj.xtc")
        self.prefix = os.path.join(self.d, "rg")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _f(self, name):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            fh.write("x")
        return p

    def run_with(self, xvg_text, check=None):
        def fake_run(cmd, log_prefix="gmx", stdin_input=None):
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write(xvg_text)
            return ""
        patches = [patch("analysis_backend.radius_of_gyration.run_gromacs_command", fake_run),
                   patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=V)]
        if check is not None:
            patches.append(patch("analysis_backend.radius_of_gyration.statistical_reality_check", check))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return run_gyrate(self.tpr, self.traj, output_prefix=self.prefix)

    def run_series(self, offset_ns=0.0, n=3000, step_ps=100.0, **kw):
        rg = synthetic_rg(n)
        times_ps = offset_ns * 1000.0 + np.arange(n) * step_ps
        return self.run_with(xvg(times_ps, rg), **kw), rg, times_ps


class TestLegendIdentification(unittest.TestCase):
    def test_total_rg_found_by_legend_not_position(self):
        self.assertEqual(_total_rg_column(["Rg", "RgX", "RgY", "RgZ"]), 0)
        self.assertEqual(_total_rg_column(["RgX", "Rg", "RgY", "RgZ"]), 1)
        self.assertEqual(_total_rg_column(["Rg(x)", "Rg", "Rg(y)"]), 1)
        self.assertEqual(_total_rg_column(["Rg_X", "RG"]), 1)
        self.assertIsNone(_total_rg_column([]))
        self.assertIsNone(_total_rg_column(["RgX", "RgY", "RgZ"]))
        self.assertIsNone(_total_rg_column(["Rg", "rg", "RgX"]))   # ambiguous -> not guessed


class TestNormal(Base):
    def test_actual_series_timestamps_and_units_passed(self):
        seen = {}

        def spy(series, **kw):
            seen["series"], seen["kw"] = np.array(series), kw
            return real_check(series, **kw)
        res, rg, times_ps = self.run_series(check=spy)
        np.testing.assert_allclose(seen["series"], rg, atol=1e-7)                 # total Rg, not X/Y/Z
        np.testing.assert_allclose(seen["kw"]["times"], times_ps / 1000.0, atol=1e-6)  # ps -> ns
        self.assertAlmostEqual(seen["kw"]["dt"], 0.1)
        self.assertEqual((seen["kw"]["time_unit"], seen["kw"]["value_unit"]), ("ns", "nm"))
        np.testing.assert_allclose(seen["series"], res.data["Rg"].to_numpy())

    def test_results_in_metadata(self):
        res, rg, _ = self.run_series()
        rc = res.metadata["statistical_reality_check"]
        self.assertIn(rc["status"], ("no_issues_detected", "inconclusive"))
        self.assertEqual(rc["series"]["identified_by"], "xvg legend")
        self.assertEqual(rc["series"]["legend"], "Rg")
        s = rc["summary"]
        eq = s["candidate_equilibration"]
        self.assertGreater(eq["t0_frame"], 0)
        self.assertAlmostEqual(eq["t0_time"], eq["t0_frame"] * 0.1, places=4)
        self.assertEqual(eq["time_unit"], "ns")
        self.assertGreater(s["ess"], 0)
        self.assertLess(s["ess"], s["analysis_segment"]["n_frames"])
        self.assertGreater(s["sem_corrected"], s["sem_naive"])
        self.assertEqual(s["value_unit"], "nm")
        self.assertIn(s["block_convergence"]["status"], ("converged", "not_converged", "insufficient_data"))
        for key in ("equilibration", "autocorrelation", "effective_sample_size", "block_averaging",
                    "cross_checks", "overall", "limitations"):
            self.assertIn(key, rc["full"])
        self.assertEqual(rc["frame_spacing"]["n_frames"], 3000)
        json.dumps(rc)

    def test_existing_outputs_preserved(self):
        res, rg, _ = self.run_series(n=400)
        m = res.metadata
        for key in ("analysis_type", "command", "group", "weighting", "time_unit", "parameters", "units",
                    "gmx_command", "time_coverage", "warnings", "selection"):
            self.assertIn(key, m)
        self.assertEqual(list(res.data.columns), ["Time (ps)", "Rg", "Rg_X", "Rg_Y", "Rg_Z"])
        self.assertEqual(set(res.statistics), {"mean", "std", "min", "max", "median"})
        self.assertAlmostEqual(res.statistics["mean"], float(np.mean(rg)), places=6)
        self.assertTrue(os.path.exists(res.output_csv) and os.path.exists(res.output_xvg))

    def test_check_warnings_surface(self):
        res, _, _ = self.run_series(n=30)
        self.assertTrue(any(w.startswith("Statistical check:") for w in res.metadata["warnings"])
                        or res.metadata["statistical_reality_check"]["status"] == "not_run")


class TestNonzeroStart(Base):
    def test_trajectory_starting_at_200_ns(self):
        res, _, _ = self.run_series(offset_ns=200.0)
        s = res.metadata["statistical_reality_check"]["summary"]
        eq = s["candidate_equilibration"]
        self.assertGreater(eq["t0_frame"], 0)
        self.assertAlmostEqual(eq["t0_time"], 200.0 + eq["t0_frame"] * 0.1, places=3)   # actual timestamp
        self.assertAlmostEqual(eq["t0_elapsed"], eq["t0_frame"] * 0.1, places=3)
        self.assertGreaterEqual(eq["t0_time"], 200.0)
        self.assertEqual(eq["time_origin"], "supplied_times")
        self.assertAlmostEqual(s["analysis_segment"]["start_time"], eq["t0_time"], places=3)
        self.assertAlmostEqual(res.metadata["statistical_reality_check"]["full"]["inputs"]["start_time"],
                               200.0, places=3)

    def test_trajectory_starting_at_zero(self):
        res, _, _ = self.run_series(offset_ns=0.0)
        eq = res.metadata["statistical_reality_check"]["summary"]["candidate_equilibration"]
        self.assertAlmostEqual(eq["t0_time"], eq["t0_elapsed"], places=4)


class TestSkipsAndFailures(Base):
    def test_irregular_sampling_skipped_with_reason(self):
        calls = []
        times = np.concatenate([np.arange(0, 2000, 100.0), np.arange(2000, 20000, 1000.0)])
        res = self.run_with(xvg(times, synthetic_rg(len(times))), check=lambda *a, **k: calls.append(1))
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(calls, [])
        self.assertEqual((rc["status"], rc["reason_code"]), ("not_run", "irregular_sampling"))
        self.assertIn("not equally spaced", rc["reason"])
        self.assertTrue(any("not run" in w for w in res.metadata["warnings"]))
        self.assertEqual(res.data.shape[0], len(times))                          # Rg result intact

    def test_insufficient_data_skipped_with_reason(self):
        calls = []
        res = self.run_with(xvg(np.arange(10) * 100.0, synthetic_rg(10)), check=lambda *a, **k: calls.append(1))
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(calls, [])
        self.assertEqual((rc["status"], rc["reason_code"]), ("not_run", "insufficient_data"))
        self.assertIn("Only 10 frames", rc["reason"])
        self.assertNotIn("summary", rc)                                            # nothing invented

    def test_no_legends_means_total_rg_not_identified_so_skipped(self):
        calls = []
        text = xvg(np.arange(300) * 100.0, synthetic_rg(300), legends="")
        res = self.run_with(text, check=lambda *a, **k: calls.append(1))
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(calls, [])
        self.assertEqual(rc["reason_code"], "no_total_rg_legend")
        self.assertEqual(res.data.shape[0], 300)                                   # positional CSV unchanged

    def test_time_axis_not_in_ps_skipped(self):
        calls = []
        text = xvg(np.arange(300) * 0.1, synthetic_rg(300), x_label="Time (ns)")
        res = self.run_with(text, check=lambda *a, **k: calls.append(1))
        self.assertEqual(calls, [])
        self.assertEqual(res.metadata["statistical_reality_check"]["reason_code"], "time_unit_unverified")

    def test_explicit_ps_label_accepted(self):
        text = xvg(np.arange(300) * 100.0, synthetic_rg(300), x_label="Time (ps)")
        res = self.run_with(text)
        self.assertNotEqual(res.metadata["statistical_reality_check"]["status"], "not_run")

    def test_check_exception_recorded_rg_still_returned(self):
        def boom(*a, **k):
            raise RuntimeError("numerical failure")
        res, rg, _ = self.run_series(n=400, check=boom)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "error")
        self.assertIn("numerical failure", rc["error"])
        self.assertNotIn("summary", rc)
        self.assertTrue(any("reality check error" in w for w in res.metadata["warnings"]))
        self.assertEqual(res.data.shape[0], 400)

    def test_gromacs_failure_never_reaches_check(self):
        called = []
        with patch("analysis_backend.radius_of_gyration.run_gromacs_command", side_effect=RuntimeError("gmx failed")), \
             patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=V), \
             patch("analysis_backend.radius_of_gyration.statistical_reality_check", lambda *a, **k: called.append(1)):
            with self.assertRaises(RuntimeError):
                run_gyrate(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertEqual(called, [])
        self.assertFalse(os.path.exists(self.prefix + ".csv"))

    def test_nan_output_fails_before_check(self):
        called = []
        text = LEGENDS + "0 nan 1 1 1\n100 2.6 1 1 1\n"

        def fake_run(cmd, log_prefix="gmx", stdin_input=None):
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write(text)
            return ""
        with patch("analysis_backend.radius_of_gyration.run_gromacs_command", fake_run), \
             patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=V), \
             patch("analysis_backend.radius_of_gyration.statistical_reality_check", lambda *a, **k: called.append(1)):
            with self.assertRaises(ValueError):
                run_gyrate(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()


class TestFormattedLegends(Base):
    def test_gromacs_subscripts_preserve_total_and_components(self):
        for separator in ("/", chr(92)):
            with self.subTest(separator=separator):
                legends = '@ s0 legend "Rg"' + chr(10)
                for i, axis in enumerate("XYZ", 1):
                    legends += f'@ s{i} legend "Rg{separator}s{axis}{separator}N"' + chr(10)
                result = self.run_with(xvg([0, 10, 20], [2., 3., 4.], legends=legends, x_label="Time (ps)"))
                np.testing.assert_allclose(result.data["Rg"], [2., 3., 4.])
                np.testing.assert_allclose(result.data["Rg_X"], [1., 1.5, 2.])
                self.assertTrue(os.path.isfile(result.output_csv))

    def test_unknown_duplicate_and_reordered_components_rejected(self):
        for labels in (
            ["Rg", "Rg/sQ/N", "Rg/sY/N", "Rg/sZ/N"],
            ["Rg", "Rg/sX/N", "Rg/sX/N", "Rg/sZ/N"],
            ["Rg", "Rg/sY/N", "Rg/sX/N", "Rg/sZ/N"],
        ):
            legends = "".join(f'@ s{i} legend "{label}"' + chr(10) for i, label in enumerate(labels))
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "legends"):
                self.run_with(xvg([0,10], [2.,3.], legends=legends))
