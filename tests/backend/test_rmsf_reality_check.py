"""Integration tests: compare_rmsf_windows / plan_time_windows inside run_rmsf.

ALL MOCKED. `subprocess.Popen` is replaced by a fake gmx that writes a synthetic per-residue RMSF .xvg
for whatever -b/-e window it is asked for. These tests verify orchestration: which windows are run,
which flags they get, that the real profiles are compared, and how problems are reported. They are NOT
validation on a real GROMACS trajectory, and they cannot show that real RMSF profiles are reproducible.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from analysis_backend.rmsf import run_rmsf

V = {"version_string": "GROMACS version:    2026.2", "major_year": 2026, "available": True}
N_RES = 120


def base_profile(n=N_RES):
    x = np.arange(n)
    return 0.10 + 0.30 * np.exp(-((x - 30) / 5.0) ** 2) + 0.50 * np.exp(-((x - 90) / 6.0) ** 2)  # nm


def xvg(ids, values):
    return ('@ xaxis  label "Residue"\n@ yaxis  label "RMSF (nm)"\n' +
            "".join(f"{i} {v:.6f}\n" for i, v in zip(ids, values)))


class FakeGmx:
    """Fake Popen factory. `profile(b_ps, e_ps)` -> (ids, values) or None (-> empty output)."""

    def __init__(self, profile, fail_on_call=None, missing_on_call=None):
        self.profile, self.fail_on_call, self.missing_on_call = profile, fail_on_call, missing_on_call
        self.cmds, self.stdins = [], []

    def __call__(self, cmd, **kwargs):
        self.cmds.append(list(cmd))
        n = len(self.cmds)
        if self.missing_on_call == n:
            raise FileNotFoundError("gmx")
        proc = MagicMock()
        proc.returncode = 1 if self.fail_on_call == n else 0

        def comm(input=None):
            self.stdins.append(input)
            if proc.returncode != 0:
                return ("", "gmx exploded")
            b = float(cmd[cmd.index("-b") + 1]) if "-b" in cmd else 0.0
            e = float(cmd[cmd.index("-e") + 1]) if "-e" in cmd else float("inf")
            out = self.profile(b, e)
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write("" if out is None else xvg(*out))
            return ("", "")
        proc.communicate.side_effect = comm
        return proc


def consistent(b_ps, e_ps):
    rng = np.random.default_rng(int(b_ps) % 100003 + 1)
    return np.arange(1, N_RES + 1), base_profile() + rng.uniform(-0.01, 0.01, N_RES)


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.tpr = self._f("topol.tpr")
        self.traj = self._f("traj.trr")
        self.prefix = os.path.join(self.d, "rmsf")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _f(self, name):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            fh.write("x")
        return p

    def go(self, fake, **kw):
        with patch("analysis_backend.rmsf.subprocess.Popen", side_effect=fake), \
             patch("analysis_backend.rmsf.get_gromacs_version_info", return_value=V):
            return run_rmsf(self.tpr, self.traj, output_prefix=self.prefix, **kw)

    @staticmethod
    def flags(cmd):
        return {k: cmd[cmd.index(k) + 1] for k in ("-b", "-e", "-dt", "-o") if k in cmd}


class TestConsistentProfiles(Base):
    def test_windows_run_and_compared(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, b=0.0, e=300.0, reliability_windows=3)
        self.assertEqual(len(fake.cmds), 4)                                    # 1 main + 3 windows
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "consistent")
        self.assertEqual([w["label"] for w in rc["windows"]], ["0-100 ns", "100-200 ns", "200-300 ns"])
        self.assertEqual(len(rc["full"]["pairwise"]), 3)
        for p in rc["full"]["pairwise"]:
            self.assertGreater(p["pearson"], 0.95)
            self.assertGreater(p["spearman"], 0.7)   # rank-based: noisy flat baseline lowers it
            self.assertLess(p["rmse"], 0.02)
        self.assertGreater(rc["summary"]["min_pearson"], 0.95)
        self.assertEqual(rc["summary"]["n_residues"], N_RES)
        self.assertTrue(rc["summary"]["independent_windows"])
        self.assertEqual(rc["full"]["rmsf_unit"], "nm")
        json.dumps(rc)

    def test_window_commands_use_same_settings_and_ps_conversion(self):
        fake = FakeGmx(consistent)
        self.go(fake, e=300.0, dt=1.0, reliability_windows=3, fit=True, res=True)
        main, windows = fake.cmds[0], fake.cmds[1:]
        for w in windows:
            for flag in ("-s", "-f"):
                self.assertEqual(w[w.index(flag) + 1], main[main.index(flag) + 1])
            self.assertIn("-res", w)
            self.assertIn("-fit", w)
            self.assertNotIn("-tu", w)
            self.assertEqual(w[w.index("-dt") + 1], "1000.0")                  # same sampling interval, in ps
        self.assertEqual(fake.stdins, ["C-alpha\n"] * 4)                       # same selection every run
        got = [(self.flags(w).get("-b"), self.flags(w)["-e"]) for w in windows]
        self.assertEqual(got, [(None, "100000.0"), ("100000.0", "200000.0"), ("200000.0", "300000.0")])
        self.assertEqual(len({w[w.index("-o") + 1] for w in windows} | {main[main.index("-o") + 1]}), 4)
        for w in windows:
            self.assertTrue(w[w.index("-o") + 1].startswith(self.prefix + "_window"))

    def test_existing_outputs_preserved(self):
        res = self.go(FakeGmx(consistent), e=300.0, reliability_windows=2)
        self.assertEqual(list(res.data.columns), ["Residue", "RMSF"])
        self.assertEqual(res.output_xvg, self.prefix + ".xvg")
        self.assertTrue(os.path.exists(res.output_csv))
        for key in ("analysis_type", "selection", "per_residue", "x_unit", "y_unit", "fit", "parameters",
                    "units", "gmx_command", "gmx_stdin_groups", "warnings"):
            self.assertIn(key, res.metadata)
        self.assertEqual(res.metadata["gmx_command"][-2:], ["-e", "300000.0"])
        self.assertEqual(set(res.statistics), {"mean", "std", "min", "max", "median"})

    def test_per_residue_differences_are_actual_differences(self):
        res = self.go(FakeGmx(consistent), e=200.0, reliability_windows=2)
        rc = res.metadata["statistical_reality_check"]
        d = rc["per_residue_differences"][0]
        self.assertEqual((d["a"], d["b"], d["unit"]), ("0-100 ns", "100-200 ns", "nm"))
        self.assertEqual(len(d["values"]), N_RES)
        self.assertEqual(d["ids"][:3], [1, 2, 3])
        a = consistent(0.0, 0)[1]
        b = consistent(100000.0, 0)[1]
        np.testing.assert_allclose(d["values"], b - a, atol=1e-5)               # b minus a, from the profiles
        self.assertEqual(len(d["largest_abs"]), 10)
        self.assertEqual(len(rc["full"]["per_residue"]["mean"]), N_RES)

    def test_three_windows_cleanup_does_not_break(self):
        for _ in range(2):                                                     # second run removes stale files
            res = self.go(FakeGmx(consistent), e=300.0, reliability_windows=3)
        self.assertEqual(res.metadata["statistical_reality_check"]["status"], "consistent")


class TestDifferingProfiles(Base):
    def test_changed_loop_is_flagged(self):
        def differing(b_ps, e_ps):
            ids, v = consistent(b_ps, e_ps)
            if b_ps >= 100000.0:                                               # loop opens only in window 2
                v = v.copy()
                v[85:100] += 0.6 * np.exp(-((np.arange(15) - 7) / 3.0) ** 2)
                v[20:40] *= 0.3
            return ids, v
        res = self.go(FakeGmx(differing), e=200.0, reliability_windows=2)
        rc = res.metadata["statistical_reality_check"]
        self.assertNotEqual(rc["status"], "consistent")
        self.assertGreater(rc["summary"]["unstable_residue_count"], 0)
        top = rc["per_residue_differences"][0]["largest_abs"][0]
        self.assertTrue(20 <= top["id"] <= 40 or 86 <= top["id"] <= 101, top)
        self.assertTrue(any("Statistical" in w or "not" in w.lower() for w in rc["warnings"]))

    def test_unrelated_profiles_inconsistent(self):
        def random_profile(b_ps, e_ps):
            rng = np.random.default_rng(int(b_ps) % 9973 + 7)
            return np.arange(1, N_RES + 1), rng.uniform(0.05, 0.6, N_RES)
        rc = self.go(FakeGmx(random_profile), e=200.0, reliability_windows=2).metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "inconsistent")


class TestNonzeroStart(Base):
    def test_windows_start_at_200_ns(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, b=200.0, e=400.0, reliability_windows=4)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual([w["label"] for w in rc["windows"]],
                         ["200-250 ns", "250-300 ns", "300-350 ns", "350-400 ns"])
        self.assertEqual([w["begin"] for w in rc["windows"]], [200.0, 250.0, 300.0, 350.0])
        self.assertEqual([w["unit"] for w in rc["windows"]], ["ns"] * 4)
        starts = [self.flags(c)["-b"] for c in fake.cmds[1:]]
        self.assertEqual(starts, ["200000.0", "250000.0", "300000.0", "350000.0"])
        self.assertEqual(rc["full"]["windows"][0]["begin"], 200.0)             # actual times reach the comparison
        self.assertEqual(rc["settings"]["span_ns"], [200.0, 400.0])
        self.assertEqual(rc["status"], "consistent")

    def test_custom_span_inside_analysis_window(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, b=200.0, e=400.0, reliability_windows=2, reliability_span=(300.0, 400.0))
        self.assertEqual([w["label"] for w in res.metadata["statistical_reality_check"]["windows"]],
                         ["300-350 ns", "350-400 ns"])

    def test_span_without_end_time_works_when_given(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, reliability_windows=2, reliability_span=(100.0, 500.0))
        self.assertEqual(res.metadata["statistical_reality_check"]["status"], "consistent")
        self.assertNotIn("-e", fake.cmds[0])                                   # main run still whole trajectory

    def test_invalid_span_raises_before_any_gmx_run(self):
        fake = FakeGmx(consistent)
        for kw in [dict(b=200.0, e=400.0, reliability_span=(100.0, 300.0)),
                   dict(b=200.0, e=400.0, reliability_span=(300.0, 500.0)),
                   dict(reliability_span=(5.0, 5.0)), dict(reliability_span="abc"),
                   dict(reliability_span=(1.0, 2.0), reliability_windows=None)]:
            kw.setdefault("reliability_windows", 2)
            with self.assertRaises(ValueError, msg=str(kw)):
                self.go(fake, **kw)
        self.assertEqual(fake.cmds, [])


class TestMismatchedResidues(Base):
    def test_different_residue_ids_inconclusive(self):
        def shifted(b_ps, e_ps):
            ids, v = consistent(b_ps, e_ps)
            return (ids + 1 if b_ps >= 100000.0 else ids), v
        res = self.go(FakeGmx(shifted), e=200.0, reliability_windows=2)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual((rc["status"], rc["reason_code"]), ("inconclusive", "residue_mapping_mismatch"))
        self.assertIn("same", rc["reason"])
        self.assertNotIn("summary", rc)                                        # no invented comparison
        self.assertNotIn("full", rc)
        self.assertTrue(any("inconclusive" in w for w in res.metadata["warnings"]))
        self.assertEqual(len(res.data), N_RES)                                 # main RMSF intact

    def test_different_profile_lengths_inconclusive(self):
        def fewer(b_ps, e_ps):
            ids, v = consistent(b_ps, e_ps)
            return (ids[:100], v[:100]) if b_ps >= 100000.0 else (ids, v)
        rc = self.go(FakeGmx(fewer), e=200.0, reliability_windows=2).metadata["statistical_reality_check"]
        self.assertEqual(rc["reason_code"], "residue_mapping_mismatch")
        self.assertIn("120", rc["reason"])
        self.assertIn("100", rc["reason"])


class TestInsufficientData(Base):
    def test_not_requested_by_default_and_no_extra_runs(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, e=200.0)
        self.assertEqual(len(fake.cmds), 1)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual((rc["status"], rc["reason_code"]), ("not_run", "not_requested"))
        self.assertFalse(any("reality check" in w for w in res.metadata["warnings"]))

    def test_end_time_unknown_means_no_windows(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, reliability_windows=2)
        self.assertEqual(len(fake.cmds), 1)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["reason_code"], "end_time_unknown")
        self.assertIn("reliability_span", rc["reason"])
        self.assertTrue(any("end time" in w for w in res.metadata["warnings"]))

    def test_too_few_residues(self):
        def tiny(b_ps, e_ps):
            return np.arange(1, 6), np.array([0.1, 0.2, 0.3, 0.2, 0.1]) + (b_ps > 0) * 0.001
        rc = self.go(FakeGmx(tiny), e=200.0, reliability_windows=2).metadata["statistical_reality_check"]
        self.assertEqual((rc["status"], rc["reason_code"]), ("insufficient_data", "insufficient_data"))
        self.assertIn("Only 5 residues", rc["reason"])
        self.assertNotIn("summary", rc)                                        # no numbers invented
        self.assertNotIn("full", rc)

    def test_windows_shorter_than_two_frames(self):
        fake = FakeGmx(consistent)
        res = self.go(fake, e=1.0, dt=0.4, reliability_windows=3)              # 0.33 ns windows, dt 0.4 ns
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual((rc["status"], rc["reason_code"]), ("inconclusive", "insufficient_data"))
        self.assertEqual(len(fake.cmds), 1)

    def test_empty_window_output(self):
        def empty_second(b_ps, e_ps):
            return None if b_ps >= 100000.0 else consistent(b_ps, e_ps)
        rc = self.go(FakeGmx(empty_second), e=200.0, reliability_windows=2).metadata["statistical_reality_check"]
        self.assertEqual((rc["status"], rc["reason_code"]), ("inconclusive", "window_output_unusable"))
        self.assertIn("100-200 ns", rc["reason"])

    def test_invalid_reliability_windows_raise_before_gmx(self):
        fake = FakeGmx(consistent)
        for bad in (1, 0, -2, 21, 2.5, "2", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.go(fake, e=200.0, reliability_windows=bad)
        self.assertEqual(fake.cmds, [])


class TestGromacsFailure(Base):
    def test_main_run_failure_raises_and_no_windows_run(self):
        fake = FakeGmx(consistent, fail_on_call=1)
        with self.assertRaises(RuntimeError):
            self.go(fake, e=200.0, reliability_windows=2)
        self.assertEqual(len(fake.cmds), 1)
        self.assertFalse(os.path.exists(self.prefix + ".csv"))

    def test_window_failure_is_reported_main_result_kept(self):
        fake = FakeGmx(consistent, fail_on_call=3)                             # second window fails
        res = self.go(fake, e=300.0, reliability_windows=3)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual((rc["status"], rc["reason_code"]), ("error", "window_run_failed"))
        self.assertIn("100-200 ns", rc["reason"])
        self.assertIn("gmx exploded", rc["error"])
        self.assertEqual(len(fake.cmds), 3)                                    # stopped at the failure
        self.assertNotIn("summary", rc)
        self.assertTrue(os.path.exists(res.output_csv))
        self.assertTrue(any("reality check error" in w for w in res.metadata["warnings"]))

    def test_gmx_missing_during_window_run(self):
        fake = FakeGmx(consistent, missing_on_call=2)
        res = self.go(fake, e=200.0, reliability_windows=2)
        rc = res.metadata["statistical_reality_check"]
        self.assertEqual(rc["status"], "error")
        self.assertIn("not found", rc["error"])

    def test_gmx_missing_for_main_run_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.go(FakeGmx(consistent, missing_on_call=1), e=200.0, reliability_windows=2)


if __name__ == "__main__":
    unittest.main()
