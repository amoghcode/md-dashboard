"""Group / selection resolution tests for gmx rms, rmsf and gyrate wrappers.

MOCKED: gmx is replaced by fakes, so these tests check what the wrappers WOULD send (stdin tokens,
-sel strings) and that invalid selections are rejected before gmx is started. They do not prove that a
real GROMACS accepts those tokens; see tests/real_gromacs_group_resolution.py (needs a real gmx).

What the GROMACS manual says (checked): classic prompts take a group number or name; selection-based
tools take `group NR` (zero-based) or `group NAME` (NAME may match PART of a name); with -n the groups
come from the index file, otherwise automatic groups are generated.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from analysis_backend.gmx_options import GroupRef, resolve_group_ref
from analysis_backend.radius_of_gyration import run_gyrate
from analysis_backend.rmsd import run_rmsd
from analysis_backend.rmsf import run_rmsf

V2026 = {"version_string": "GROMACS version:    2026.2", "major_year": 2026, "available": True}
V2023 = {"version_string": "GROMACS version:    2023.3", "major_year": 2023, "available": True}

# numbering GROMACS shows: 0 System, 1 Protein, 2 C-alpha, 3 Backbone, 4 PRO_A, 5 PROA, 6 JZ4, 7 Water, 8 Empty
INDEX = ("[ System ]\n1 2 3 4 5 6 7 8\n[ Protein ]\n1 2 3 4\n[ C-alpha ]\n2 4\n[ Backbone ]\n1 2 3\n"
         "[ PRO_A ]\n1 2\n[ PROA ]\n3 4\n[ JZ4 ]\n5 6\n[ Water ]\n7 8\n[ Empty ]\n")
SHIFTED = "[ Dup ]\n1\n[ Dup ]\n2\n[ Protein ]\n1 2\n[ Ligand One ]\n3\n[ 7 ]\n4\n"

RMSD_XVG = '@ xaxis  label "Time (ns)"\n0.0 0.0\n0.5 0.1\n1.0 0.2\n'
RMSF_XVG = "".join(f"{i} {0.1 + i * 0.001}\n" for i in range(1, 31))
RG_XVG = ('@ s0 legend "Rg"\n@ s1 legend "RgX"\n@ s2 legend "RgY"\n@ s3 legend "RgZ"\n'
          "0 2.6 2 2 2\n500 2.7 2 2 2\n1000 2.6 2 2 2\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.tpr, self.traj = self._f("t.tpr"), self._f("t.xtc")
        self.ndx = self._f("index.ndx", INDEX)
        self.prefix = os.path.join(self.d, "out")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _f(self, name, content="x"):
        p = os.path.join(self.d, name)
        with open(p, "w") as fh:
            fh.write(content)
        return p

    def gmx_run(self, xvg):
        calls = []

        def fake(cmd, log_prefix="gmx", stdin_input=None):
            calls.append((list(cmd), stdin_input))
            with open(cmd[cmd.index("-o") + 1], "w") as fh:
                fh.write(xvg)
            return ""
        fake.calls = calls
        return fake

    def rmsd(self, **kw):
        fake = self.gmx_run(RMSD_XVG)
        with patch("analysis_backend.rmsd.run_gromacs_command", fake), \
             patch("analysis_backend.rmsd.get_gromacs_version_info", return_value=V2026):
            res = run_rmsd(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        return res, fake.calls[0]

    def rmsf(self, **kw):
        cap = {}

        def popen(cmd, **k):
            cap.setdefault("cmds", []).append(list(cmd))
            proc = MagicMock(returncode=0)

            def comm(input=None):
                cap["stdin"] = input
                with open(cmd[cmd.index("-o") + 1], "w") as fh:
                    fh.write(RMSF_XVG)
                return ("", "")
            proc.communicate.side_effect = comm
            return proc
        with patch("analysis_backend.rmsf.subprocess.Popen", side_effect=popen), \
             patch("analysis_backend.rmsf.get_gromacs_version_info", return_value=V2026):
            res = run_rmsf(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        return res, cap

    def gyrate(self, version=V2026, **kw):
        fake = self.gmx_run(RG_XVG)
        with patch("analysis_backend.radius_of_gyration.run_gromacs_command", fake), \
             patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=version):
            res = run_gyrate(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        return res, fake.calls[0]


class TestGroupRef(Base):
    def test_numbers_follow_file_order_case_insensitively(self):
        expect = {"system": 0, "PROTEIN": 1, "c-alpha": 2, "Backbone": 3, "pro_a": 4, "proa": 5, "jz4": 6}
        for name, num in expect.items():
            ref = resolve_group_ref(name, self.ndx)
            self.assertEqual(ref.number, num, name)
            self.assertEqual(ref.source, "index_file")
            self.assertEqual(ref.stdin_token, str(num))
            self.assertEqual(ref.selection_expr, f"group {num}")
        self.assertEqual(resolve_group_ref("proa", self.ndx).name, "PROA")        # file's own spelling

    def test_similar_names_stay_distinct(self):
        # GROMACS' prompt ignores '-' and '_' when comparing names, so PRO_A / PROA could be confused by
        # name; their numbers are unambiguous.
        self.assertNotEqual(resolve_group_ref("PRO_A", self.ndx).stdin_token,
                            resolve_group_ref("PROA", self.ndx).stdin_token)

    def test_duplicate_headers_before_the_group_shift_numbering(self):
        ndx = self._f("shift.ndx", SHIFTED)
        self.assertEqual(resolve_group_ref("Protein", ndx).number, 2)             # Dup, Dup, Protein
        self.assertEqual(resolve_group_ref("ligand one", ndx).number, 3)
        self.assertEqual(resolve_group_ref("7", ndx).number, 4)                   # numeric-looking name -> its position
        with self.assertRaises(ValueError):
            resolve_group_ref("Dup", ndx)                                         # ambiguous name itself

    def test_automatic_groups_use_names_without_index(self):
        for given, canon in [("backbone", "Backbone"), ("C-ALPHA", "C-alpha"), ("protein", "Protein"),
                             ("non-protein", "non-Protein")]:
            ref = resolve_group_ref(given, None)
            self.assertEqual(ref, GroupRef(canon, None, "default_groups"))
            self.assertEqual(ref.stdin_token, canon)
            self.assertEqual(ref.selection_expr, f'group "{canon}"')

    def test_invalid_selections_rejected(self):
        bad = [("Nope", self.ndx), ("Empty", self.ndx), ("", self.ndx), ("  ", None),
               ("PROA", None), ("JZ4", None), ("Bogus", None),
               ("Protein\n0", self.ndx), ('Pro"tein', None), ("Pro\rtein", None)]
        for name, ndx in bad:
            with self.assertRaises(ValueError, msg=repr((name, ndx))):
                resolve_group_ref(name, ndx)
        with self.assertRaises(ValueError):
            resolve_group_ref("SOL", self.ndx)                                     # automatic name absent from this file
        self.assertEqual(resolve_group_ref(" Protein\r\n", None).name, "Protein")   # surrounding whitespace is stripped
        absent = self._f("small.ndx", "[ Protein ]\n1 2\n")
        with self.assertRaises(ValueError):
            resolve_group_ref("Backbone", absent)                                  # with -n only the file's groups exist
        with self.assertRaises(ValueError):
            resolve_group_ref(None, None)
        with self.assertRaises(ValueError):
            resolve_group_ref("A", self._f("bad.ndx", "1 2 3\n[ A ]\n1\n"))        # atoms before any header
        with self.assertRaises(ValueError):
            resolve_group_ref("A", self._f("bad2.ndx", "[ A ]\n1 two 3\n"))
        with self.assertRaises(FileNotFoundError):
            resolve_group_ref("A", os.path.join(self.d, "missing.ndx"))


class TestRMSDGroups(Base):
    def test_index_file_sends_numbers_fit_then_calc(self):
        res, (cmd, stdin) = self.rmsd(index_file=self.ndx, fit_group="Backbone", calc_group="c-alpha")
        self.assertEqual(stdin, "3\n2\n")
        m = res.metadata
        self.assertEqual((m["fit_group"], m["calc_group"]), ("Backbone", "C-alpha"))
        self.assertEqual(m["gmx_stdin_input"], ["3", "2"])
        self.assertEqual(m["gmx_stdin_groups"], ["Backbone", "C-alpha"])
        self.assertEqual(m["group_resolution"]["calc_group"],
                         {"name": "C-alpha", "index_number": 2, "source": "index_file", "sent_as": "number"})

    def test_fit_none_answers_both_prompts_with_calc_number(self):
        _, (_, stdin) = self.rmsd(index_file=self.ndx, fit_mode="none", calc_group="PROA")
        self.assertEqual(stdin, "5\n5\n")

    def test_no_index_sends_canonical_names(self):
        res, (_, stdin) = self.rmsd(fit_group="backbone", calc_group="PROTEIN")
        self.assertEqual(stdin, "Backbone\nProtein\n")
        self.assertEqual(res.metadata["group_resolution"]["fit_group"]["sent_as"], "name")

    def test_invalid_selection_never_starts_gmx(self):
        fake = self.gmx_run(RMSD_XVG)
        cases = [dict(index_file=self.ndx, calc_group="Nope"), dict(index_file=self.ndx, fit_group="Empty"),
                 dict(calc_group="PROA"), dict(fit_group=""), dict(index_file=self.ndx, calc_group="SOL")]
        with patch("analysis_backend.rmsd.run_gromacs_command", fake):
            for kw in cases:
                with self.assertRaises(ValueError, msg=str(kw)):
                    run_rmsd(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        self.assertEqual(fake.calls, [])


class TestRMSFGroups(Base):
    def test_index_file_sends_number(self):
        res, cap = self.rmsf(index_file=self.ndx, selection="proa")
        self.assertEqual(cap["stdin"], "5\n")
        self.assertEqual(res.metadata["selection"], "PROA")
        self.assertEqual(res.metadata["gmx_stdin_input"], ["5"])
        self.assertEqual(res.metadata["group_resolution"]["selection"]["index_number"], 5)

    def test_default_selection_without_index_is_c_alpha_name(self):
        res, cap = self.rmsf()
        self.assertEqual(cap["stdin"], "C-alpha\n")
        self.assertEqual(res.metadata["group_resolution"]["selection"]["sent_as"], "name")

    def test_default_selection_with_index_lacking_it_is_rejected(self):
        small = self._f("small.ndx", "[ Protein ]\n1 2\n")
        with patch("analysis_backend.rmsf.subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                run_rmsf(self.tpr, self.traj, index_file=small, output_prefix=self.prefix)
            popen.assert_not_called()

    def test_window_runs_reuse_the_same_group_token(self):
        res, cap = self.rmsf(index_file=self.ndx, selection="JZ4", e=100.0,
                             reliability_windows=2)
        self.assertEqual(len(cap["cmds"]), 3)
        self.assertEqual(cap["stdin"], "6\n")

    def test_invalid_selections_never_start_gmx(self):
        for kw in [dict(index_file=self.ndx, selection="Nope"), dict(selection="PROA"),
                   dict(index_file=self.ndx, selection="Empty"), dict(selection="a\nb")]:
            with patch("analysis_backend.rmsf.subprocess.Popen") as popen:
                with self.assertRaises(ValueError, msg=str(kw)):
                    run_rmsf(self.tpr, self.traj, output_prefix=self.prefix, **kw)
                popen.assert_not_called()


class TestGyrateGroups(Base):
    def test_selection_tool_uses_group_numbers_with_index(self):
        res, (cmd, stdin) = self.gyrate(index_file=self.ndx, group="PROA")
        self.assertEqual(cmd[cmd.index("-sel") + 1], "group 5")
        self.assertEqual(stdin, "")
        self.assertEqual(res.metadata["selection_readable"], 'group "PROA"')
        self.assertEqual(res.metadata["group_resolution"]["group"]["index_number"], 5)
        self.assertIsNone(res.metadata["group_resolution"]["ligand_group"])

    def test_selection_tool_uses_quoted_name_without_index(self):
        res, (cmd, _) = self.gyrate()
        self.assertEqual(cmd[cmd.index("-sel") + 1], 'group "Protein"')
        self.assertEqual(res.metadata["group_resolution"]["group"]["sent_as"], "name")

    def test_protein_plus_ligand_numbers(self):
        res, (cmd, _) = self.gyrate(index_file=self.ndx, ligand_group="jz4")
        self.assertEqual(cmd[cmd.index("-sel") + 1], "group 1 or group 6")
        self.assertEqual(res.metadata["group_resolution"]["ligand_group"]["index_number"], 6)
        self.assertEqual(res.metadata["selection_readable"], 'group "Protein" or group "JZ4"')

    def test_legacy_prompt_gets_number_or_name(self):
        _, (cmd, stdin) = self.gyrate(version=V2023, index_file=self.ndx, group="Protein")
        self.assertEqual(stdin, "1\n")
        self.assertNotIn("-sel", cmd)
        res, (_, stdin) = self.gyrate(version=V2023, group="Protein")
        self.assertEqual(stdin, "Protein\n")
        self.assertEqual(res.metadata["gmx_stdin_input"], ["Protein"])

    def test_invalid_selections_never_start_gmx(self):
        fake = self.gmx_run(RG_XVG)
        cases = [dict(group="Nope", index_file=self.ndx), dict(group="PROA"),
                 dict(group="Protein", index_file=self.ndx, ligand_group="Water"),   # solvent is not a ligand
                 dict(group="Protein", index_file=self.ndx, ligand_group="PROA"),    # overlaps Protein
                 dict(group="Protein", index_file=self.ndx, ligand_group="Empty"),
                 dict(group="Protein", ligand_group="JZ4"),                          # ligand needs an index file
                 dict(group="Back\nbone"), dict(group='x" or group "Protein', index_file=self.ndx)]
        for version in (V2026, V2023):
            with patch("analysis_backend.radius_of_gyration.run_gromacs_command", fake), \
                 patch("analysis_backend.radius_of_gyration.get_gromacs_version_info", return_value=version):
                for kw in cases:
                    with self.assertRaises(ValueError, msg=f"{version['major_year']} {kw}"):
                        run_gyrate(self.tpr, self.traj, output_prefix=self.prefix, **kw)
        self.assertEqual(fake.calls, [])

    def test_selection_injection_is_impossible(self):
        # a group name cannot smuggle extra selection text into -sel: quotes/newlines are refused and,
        # with an index file, only a validated number is placed in the expression.
        ndx = self._f("inj.ndx", '[ Protein ]\n1\n[ x" or group "System ]\n2\n')
        res, (cmd, _) = self.gyrate(index_file=ndx, group="Protein")
        self.assertEqual(cmd[cmd.index("-sel") + 1], "group 0")
        with self.assertRaises(ValueError):
            self.gyrate(index_file=ndx, group='x" or group "System')


if __name__ == "__main__":
    unittest.main()
