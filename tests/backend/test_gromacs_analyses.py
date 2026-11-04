"""Focused tests for gmx rms / rmsf / gyrate wrappers.

IMPORTANT: every test here is MOCKED - gmx is replaced by a fake that writes a
canned .xvg. They verify command lines, stdin, unit conversion, validation and
failure handling. They do NOT prove GROMACS accepts those flags; that is covered
only by tests/real_gromacs_validation.py, which needs a real `gmx` install.
"""
import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from analysis_backend.gmx_options import (
    fmt_time, normalize_time_window, read_index_groups, resolve_group, check_ligand_group,
)
from analysis_backend.rmsd import run_rmsd
from analysis_backend.rmsf import run_rmsf
from analysis_backend.radius_of_gyration import run_gyrate

V2026 = {"version_string": "GROMACS version:    2026.2", "major_year": 2026, "available": True}
V2023 = {"version_string": "GROMACS version:    2023.3", "major_year": 2023, "available": True}

RMSD_XVG = '@ xaxis  label "Time (ns)"\n@ yaxis  label "RMSD (nm)"\n0.0 0.0\n0.5 0.12\n1.0 0.15\n'
RMSF_XVG = '@ xaxis  label "Residue"\n@ yaxis  label "RMSF (nm)"\n1 0.1\n2 0.2\n3 0.3\n'
RG_LEGENDS = '@ s0 legend "Rg"\n@ s1 legend "RgX"\n@ s2 legend "RgY"\n@ s3 legend "RgZ"\n'
RG_DATA = "0 2.6 2.0 2.1 2.2\n500 2.7 2.0 2.1 2.3\n1000 2.65 2.0 2.1 2.2\n"

INDEX = (
    "[ System ]\n1 2 3 4 5 6\n[ Protein ]\n1 2 3 4\n[ C-alpha ]\n2 4\n[ PROA ]\n1 2\n"
    "[ PROB ]\n3 4\n[ JZ4 ]\n5 6\n[ Water ]\n7 8\n"
)


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.tpr = self._f("topol.tpr")
        self.traj = self._f("traj.xtc")
        self.ndx = self._f("index.ndx", INDEX)
        self.prefix = os.path.join(self.d, "out")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _f(self, name, content="x"):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            fh.write(content)
        return p

    def fake_gmx(self, xvg_text):
        """run_gromacs_command replacement: records call, writes the xvg given by -o."""
        calls = []

        def _run(cmd, log_prefix="gmx", stdin_input=None):
            calls.append((list(cmd), stdin_input))
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write(xvg_text)
            return ""
        _run.calls = calls
        return _run


class TestOptionHelpers(Base):
    def test_time_formatting_is_exact(self):
        self.assertEqual(fmt_time(1.1, "ps"), "1100.0")
        self.assertEqual(fmt_time(0.1, "ps"), "100.0")
        self.assertEqual(fmt_time(250, "ps"), "250000.0")
        self.assertEqual(fmt_time(1.5, "ns"), "1.5")

    def test_window_validation(self):
        w = normalize_time_window(1.0, 10.0, 0.1)
        self.assertEqual((w.begin_ns, w.end_ns, w.step_ns), (1.0, 10.0, 0.1))
        w = normalize_time_window()  # defaults = whole trajectory
        self.assertIsNone(w.end_ns)
        self.assertIsNone(w.step_ns)
        for bad in [dict(b=-1), dict(b=5, e=5), dict(b=5, e=2), dict(e=0.0), dict(dt=-0.1),
                    dict(b=float("nan")), dict(e=float("inf")), dict(b="1"), dict(b=True),
                    dict(b=0, e=1, dt=2)]:
            with self.assertRaises(ValueError, msg=str(bad)):
                normalize_time_window(**bad)

    def test_index_group_resolution_is_strict_with_index(self):
        self.assertEqual(resolve_group("proa", self.ndx), "PROA")
        with self.assertRaises(ValueError):  # Backbone is NOT in this file -> gmx would fail
            resolve_group("Backbone", self.ndx)
        self.assertEqual(resolve_group("backbone", None), "Backbone")  # standard group, no -n
        with self.assertRaises(ValueError):
            resolve_group("PROA", None)  # custom needs an index file
        for bad in ["", "  ", "A\nB", 'A"B']:
            with self.assertRaises(ValueError):
                resolve_group(bad, self.ndx)

    def test_group_policy_index_vs_no_index(self):
        from analysis_backend.gmx_options import resolve_selection_group
        for fn in (resolve_group, resolve_selection_group):
            # index file supplied: ONLY groups defined in it (even standard names are rejected if absent)
            self.assertEqual(fn("protein", self.ndx), "Protein")
            for absent in ["Backbone", "SOL", "NotThere"]:
                with self.assertRaises(ValueError, msg=absent):
                    fn(absent, self.ndx)
            # no index file: ONLY automatic groups, canonical spelling; custom names rejected
            self.assertEqual(fn("non-protein", None), "non-Protein")
            self.assertEqual(fn("c-ALPHA", None), "C-alpha")
            for custom in ["PROA", "JZ4", "Bogus"]:
                with self.assertRaises(ValueError, msg=custom):
                    fn(custom, None)

    def test_duplicate_and_empty_index_groups_rejected(self):
        dup = self._f("dup.ndx", "[ A ]\n1\n[ A ]\n2\n[ Empty ]\n")
        with self.assertRaises(ValueError):
            resolve_group("A", dup)
        with self.assertRaises(ValueError):
            resolve_group("Empty", dup)
        self.assertEqual(list(read_index_groups(self.ndx))[:3], ["System", "Protein", "C-alpha"])

    def test_ligand_group_validation(self):
        self.assertTrue(isinstance(check_ligand_group("JZ4", "Protein", self.ndx), list))
        with self.assertRaises(ValueError):
            check_ligand_group("Water", "Protein", self.ndx)   # solvent, not a ligand
        with self.assertRaises(ValueError):
            check_ligand_group("PROA", "Protein", self.ndx)    # overlaps Protein
        with self.assertRaises(ValueError):
            check_ligand_group("JZ4", "Protein", None)         # needs index
        with self.assertRaises(ValueError):
            check_ligand_group("NOPE", "Protein", self.ndx)


class TestRMSD(Base):
    def run_it(self, **kw):
        fake = self.fake_gmx(RMSD_XVG)
        with patch("analysis_backend.rmsd.run_gromacs_command", fake), \
             patch("analysis_backend.rmsd.get_gromacs_version_info", return_value=V2026):
            res = run_rmsd(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        return res, fake.calls[0]

    def test_defaults(self):
        res, (cmd, stdin) = self.run_it()
        self.assertEqual(cmd, ["gmx", "rms", "-s", self.tpr, "-f", self.traj, "-o", self.prefix + ".xvg",
                               "-tu", "ns", "-mw", "-fit", "rot+trans", "-pbc"])
        self.assertEqual(stdin, "Backbone\nBackbone\n")
        m = res.metadata
        self.assertEqual(m["ref_structure"], self.tpr)
        self.assertEqual(m["units"]["time"], "ns")
        self.assertEqual(m["parameters"]["time_window"], {"begin": 0.0, "end": None, "step": None, "unit": "ns"})
        self.assertTrue(os.path.exists(res.output_csv) and os.path.exists(res.output_xvg))
        self.assertEqual(res.data.shape, (3, 2))

    def test_custom_options_and_time_in_ns(self):
        ref = self._f("ref.pdb")
        res, (cmd, stdin) = self.run_it(
            index_file=self.ndx, fit_group="PROA", calc_group="C-alpha", ref_structure=ref, mass=False,
            fit_mode="translation", pbc=False, b=1.5, e=10.0, dt=0.1)
        self.assertEqual(cmd[cmd.index("-s") + 1], ref)
        self.assertIn("-nomw", cmd)
        self.assertEqual(cmd[cmd.index("-fit") + 1], "translation")
        self.assertIn("-nopbc", cmd)
        self.assertEqual(cmd[cmd.index("-n") + 1], self.ndx)
        # -tu ns is passed, so -b/-e/-dt are given in ns UNCONVERTED
        self.assertEqual(cmd[cmd.index("-tu") + 1], "ns")
        self.assertEqual((cmd[cmd.index("-b") + 1], cmd[cmd.index("-e") + 1], cmd[cmd.index("-dt") + 1]),
                         ("1.5", "10.0", "0.1"))
        self.assertEqual(stdin, "3\n2\n")   # index file given: group NUMBERS (PROA=3, C-alpha=2)
        self.assertTrue(any("non-.tpr" in w for w in res.metadata["warnings"]))

    def test_fit_none_answers_both_prompts_with_calc_group(self):
        res, (cmd, stdin) = self.run_it(fit_mode="none", calc_group="Protein")
        self.assertEqual(stdin, "Protein\nProtein\n")
        self.assertFalse(res.metadata["parameters"]["fit_group_used"])

    def test_invalid_values_raise_before_running(self):
        fake = self.fake_gmx(RMSD_XVG)
        bad_calls = [
            dict(fit_mode="rotxy+transxy"), dict(fit_mode="progressive"), dict(fit_mode=None),
            dict(mass="yes"), dict(pbc=1), dict(fit_group="Nope", index_file=self.ndx),
            dict(calc_group="Backbone", index_file=self.ndx),  # not in the index file
            dict(b=5, e=1), dict(dt=-1),
            dict(ref_structure=os.path.join(self.d, "missing.pdb")),
            dict(ref_structure=self._f("ref.txt")),
        ]
        with patch("analysis_backend.rmsd.run_gromacs_command", fake):
            for kw in bad_calls:
                with self.assertRaises((ValueError, FileNotFoundError), msg=str(kw)):
                    run_rmsd(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        self.assertEqual(fake.calls, [])

    def test_validation_failure_keeps_previous_results(self):
        self._f("out.csv", "old")
        with self.assertRaises(ValueError):
            run_rmsd(self.tpr, self.traj, fit_mode="bogus", output_prefix=self.prefix)
        self.assertTrue(os.path.exists(self.prefix + ".csv"))

    def test_gromacs_failure_propagates_no_fake_result(self):
        with patch("analysis_backend.rmsd.run_gromacs_command", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                run_rmsd(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertFalse(os.path.exists(self.prefix + ".csv"))

    def test_missing_or_nan_output_fails(self):
        with patch("analysis_backend.rmsd.run_gromacs_command", return_value=""):
            with self.assertRaises(FileNotFoundError):
                run_rmsd(self.tpr, self.traj, output_prefix=self.prefix)
        fake = self.fake_gmx('@ xaxis  label "Time (ns)"\n0.0 nan\n1.0 0.1\n')
        with patch("analysis_backend.rmsd.run_gromacs_command", fake):
            with self.assertRaises(ValueError):
                run_rmsd(self.tpr, self.traj, output_prefix=self.prefix)

    def test_truncated_read_is_flagged(self):
        res, _ = self.run_it(e=100.0)  # canned output ends at 1 ns
        self.assertTrue(any("ends at" in w for w in res.metadata["warnings"]))
        self.assertEqual(res.metadata["time_coverage"]["last"], 1.0)


class TestRMSF(Base):
    def run_it(self, xvg=RMSF_XVG, **kw):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            proc = MagicMock()
            proc.returncode = 0

            def comm(input=None):
                captured["stdin"] = input
                with open(cmd[cmd.index("-o") + 1], "w") as fh:
                    fh.write(xvg)
                return ("", "")
            proc.communicate.side_effect = comm
            return proc
        # subprocess is one shared module: also stub the version probe so only the rmsf call is captured
        with patch("analysis_backend.rmsf.subprocess.Popen", side_effect=fake_popen), \
             patch("analysis_backend.rmsf.get_gromacs_version_info", return_value=V2026):
            res = run_rmsf(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        return res, captured

    def test_defaults(self):
        res, c = self.run_it()
        self.assertEqual(c["cmd"], ["gmx", "rmsf", "-s", self.tpr, "-f", self.traj, "-o", self.prefix + ".xvg",
                                    "-res", "-fit"])
        self.assertEqual(c["stdin"], "C-alpha\n")
        self.assertEqual(list(res.data.columns), ["Residue", "RMSF"])
        self.assertEqual(res.metadata["units"]["rmsf"], "nm")
        self.assertEqual(res.metadata["parameters"]["selection"], "C-alpha")
        self.assertTrue(any("PBC" in w for w in res.metadata["warnings"]))

    def test_time_is_converted_ns_to_ps_and_no_tu_flag(self):
        # gmx rmsf has no -tu: passing ns numbers would silently be read as ps.
        res, c = self.run_it(b=1.0, e=10.0, dt=0.1)
        cmd = c["cmd"]
        self.assertNotIn("-tu", cmd)
        self.assertEqual((cmd[cmd.index("-b") + 1], cmd[cmd.index("-e") + 1], cmd[cmd.index("-dt") + 1]),
                         ("1000.0", "10000.0", "100.0"))
        self.assertIn("ps", res.metadata["units"]["time_passed_to_gmx"])

    def test_custom_options(self):
        res, c = self.run_it(index_file=self.ndx, selection="PROB", res=False, fit=False,
                             xvg='1 0.1\n2 0.2\n')
        self.assertIn("-nores", c["cmd"])
        self.assertIn("-nofit", c["cmd"])
        self.assertEqual(c["cmd"][c["cmd"].index("-n") + 1], self.ndx)
        self.assertEqual(c["stdin"], "4\n")   # PROB is group 4 in the index file
        self.assertEqual(list(res.data.columns), ["Atom", "RMSF"])  # not mislabeled "Residue"
        self.assertTrue(any("fit=False" in w for w in res.metadata["warnings"]))
        self.assertIsNone(res.metadata["parameters"]["fit_group"])

    def test_backbone_and_protein_without_index(self):
        for g in ["Backbone", "Protein"]:
            _, c = self.run_it(selection=g)
            self.assertEqual(c["stdin"], g + "\n")

    def test_invalid_options(self):
        for kw in [dict(selection="Nope", index_file=self.ndx), dict(selection="Backbone", index_file=self.ndx),
                   dict(selection="PROA"), dict(res="yes"), dict(fit=None), dict(b=3, e=3), dict(e=0.0)]:
            with self.assertRaises(ValueError, msg=str(kw)):
                run_rmsf(self.tpr, self.traj, output_prefix=self.prefix, **kw)

    def test_nonzero_exit_raises(self):
        proc = MagicMock(returncode=1)
        proc.communicate.return_value = ("out", "err")
        with patch("analysis_backend.rmsf.subprocess.Popen", return_value=proc):
            with self.assertRaises(RuntimeError):
                run_rmsf(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertFalse(os.path.exists(self.prefix + ".csv"))

    def test_gmx_missing_raises(self):
        with patch("analysis_backend.rmsf.subprocess.Popen", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                run_rmsf(self.tpr, self.traj, output_prefix=self.prefix)

    def test_unexpected_columns_fail(self):
        with self.assertRaises(ValueError):
            self.run_it(xvg="1 0.1 0.2\n2 0.1 0.2\n")


class TestGyrate(Base):
    def run_it(self, version=V2026, xvg=RG_LEGENDS + RG_DATA, index=False, **kw):
        fake = self.fake_gmx(xvg)
        if index:
            kw["index_file"] = self.ndx
        with patch("analysis_backend.radius_of_gyration.run_gromacs_command", fake), \
             patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=version):
            res = run_gyrate(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        return res, fake.calls[0]

    def test_defaults_new_tool(self):
        res, (cmd, stdin) = self.run_it()
        self.assertEqual(cmd, ["gmx", "gyrate", "-s", self.tpr, "-f", self.traj, "-o", self.prefix + ".xvg",
                               "-sel", 'group "Protein"', "-mode", "mass", "-rmpbc"])
        self.assertEqual(stdin, "")
        self.assertEqual(list(res.data.columns), ["Time (ps)", "Rg", "Rg_X", "Rg_Y", "Rg_Z"])
        m = res.metadata
        self.assertTrue(m["mass_weighted"])
        self.assertEqual(m["units"]["rg"], "nm")
        self.assertEqual(m["time_coverage"]["last"], 1.0)  # 1000 ps -> 1 ns
        self.assertEqual(m["gyrate_implementation"], "selection-based (2024+)")

    def test_time_ns_to_ps(self):
        _, (cmd, _) = self.run_it(b=1.0, e=10.0, dt=0.1)
        self.assertEqual((cmd[cmd.index("-b") + 1], cmd[cmd.index("-e") + 1], cmd[cmd.index("-dt") + 1]),
                         ("1000.0", "10000.0", "100.0"))
        self.assertNotIn("-tu", cmd)

    def test_weighting_modes_and_pbc(self):
        for w in ["mass", "charge", "geometry"]:
            _, (cmd, _) = self.run_it(weighting=w)
            self.assertEqual(cmd[cmd.index("-mode") + 1], w)
        res, (cmd, _) = self.run_it(mass=False)  # legacy bool maps to geometry
        self.assertEqual(cmd[cmd.index("-mode") + 1], "geometry")
        self.assertFalse(res.metadata["mass_weighted"])
        _, (cmd, _) = self.run_it(pbc=False)
        self.assertIn("-normpbc", cmd)
        with self.assertRaises(ValueError):
            self.run_it(weighting="mass", mass=False)
        with self.assertRaises(ValueError):
            self.run_it(weighting="unit")

    def test_chain_and_ligand_selection(self):
        _, (cmd, _) = self.run_it(index=True, group="PROA")
        self.assertEqual(cmd[cmd.index("-sel") + 1], "group 3")
        res, (cmd, _) = self.run_it(index=True, ligand_group="JZ4")  # Protein is an automatic group
        self.assertEqual(cmd[cmd.index("-sel") + 1], "group 1 or group 5")
        self.assertEqual(res.metadata["selection_readable"], 'group "Protein" or group "JZ4"')
        self.assertEqual(res.metadata["ligand_group"], "JZ4")
        for kw in [dict(ligand_group="Water"), dict(ligand_group="PROA", group="Protein"), dict(ligand_group="X")]:
            with self.assertRaises(ValueError, msg=str(kw)):
                self.run_it(index=True, **kw)
        with self.assertRaises(ValueError):
            self.run_it(ligand_group="JZ4")  # no index file
        with self.assertRaises(ValueError):
            self.run_it(index=True, group="Nope")
        with self.assertRaises(ValueError):  # index supplied: automatic-only name absent from it is rejected
            self.run_it(index=True, group="Backbone")
        with self.assertRaises(ValueError):  # no index: custom chain group rejected
            self.run_it(group="PROA")

    def test_components_filter_csv_only(self):
        res, _ = self.run_it(components=("z", "x"))
        self.assertEqual(list(res.data.columns), ["Time (ps)", "Rg", "Rg_Z", "Rg_X"])
        with open(res.output_xvg) as fh:  # raw gmx output untouched
            self.assertEqual(len(fh.read().splitlines()[-1].split()), 5)
        res, _ = self.run_it(components=())
        self.assertEqual(list(res.data.columns), ["Time (ps)", "Rg"])
        for bad in ["xyz", ("w",), ("x", "x"), None]:
            with self.assertRaises(ValueError, msg=str(bad)):
                self.run_it(components=bad)

    def test_legacy_gromacs(self):
        _, (cmd, stdin) = self.run_it(version=V2023, group="Protein", weighting="charge")
        self.assertEqual(cmd, ["gmx", "gyrate", "-s", self.tpr, "-f", self.traj, "-o", self.prefix + ".xvg", "-q"])
        self.assertEqual(stdin, "Protein\n")
        _, (cmd, _) = self.run_it(version=V2023)
        self.assertNotIn("-mode", cmd)
        self.assertNotIn("-normpbc", cmd)
        self.assertNotIn("-rmpbc", cmd)
        for kw in [dict(mass=False), dict(weighting="geometry"), dict(pbc=False), dict(ligand_group="JZ4", index=True)]:
            with self.assertRaises(ValueError, msg=str(kw)):
                self.run_it(version=V2023, **kw)
        with self.assertRaises(ValueError):  # legacy asks the prompt: group must be IN the index file
            self.run_it(version=V2023, index=True, group="Backbone")

    def test_unknown_or_missing_gromacs(self):
        with self.assertRaises(ValueError):
            self.run_it(version={"version_string": "unknown", "major_year": None, "available": True})
        with self.assertRaises(FileNotFoundError):
            self.run_it(version={"version_string": "GROMACS Not Available", "major_year": None, "available": False})

    def test_output_sanity(self):
        with self.assertRaises(ValueError):  # wrong legend order must not be labelled by position
            self.run_it(xvg='@ s0 legend "RgX"\n@ s1 legend "Rg"\n@ s2 legend "RgY"\n@ s3 legend "RgZ"\n' + RG_DATA)
        with self.assertRaises(ValueError):
            self.run_it(xvg="0 2.6\n500 2.7\n")  # wrong column count
        res, _ = self.run_it(xvg=RG_DATA)  # no legends: positional, but flagged
        self.assertTrue(any("legends" in w for w in res.metadata["warnings"]))

    def test_gromacs_failure_propagates(self):
        with patch("analysis_backend.radius_of_gyration.run_gromacs_command", side_effect=RuntimeError("x")), \
             patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=V2026):
            with self.assertRaises(RuntimeError):
                run_gyrate(self.tpr, self.traj, output_prefix=self.prefix)
        self.assertFalse(os.path.exists(self.prefix + ".csv"))


if __name__ == "__main__":
    unittest.main()
