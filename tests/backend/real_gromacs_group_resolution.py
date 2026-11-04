"""REAL-GROMACS validation of atom-group resolution (not mocked).

Skipped unless `gmx` is on PATH and WT_RUNDIR is set. NOT run in the sandbox where this was written
(no GROMACS installed), so every claim below is still unverified until you run it:

    WT_RUNDIR=/path/to/run_directory python -m unittest tests.real_gromacs_group_resolution -v

Optional: RG_TPR (prod1.tpr), RG_TRAJ (prod1.part0017.trr), RG_NDX (index.ndx).
Outputs go to a temp directory; nothing is written into run_directory.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from analysis_backend.analysis_utils import get_gromacs_version_info
from analysis_backend.gmx_options import read_index_groups, resolve_group_ref
from analysis_backend.radius_of_gyration import run_gyrate
from analysis_backend.rmsf import run_rmsf

RUNDIR = os.environ.get("WT_RUNDIR")
VINFO = get_gromacs_version_info()


@unittest.skipUnless(VINFO["available"] and RUNDIR, "needs real gmx on PATH and WT_RUNDIR")
class RealGroupResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tpr = os.path.join(RUNDIR, os.environ.get("RG_TPR", "prod1.tpr"))
        self.traj = os.path.join(RUNDIR, os.environ.get("RG_TRAJ", "prod1.part0017.trr"))
        self.ndx = os.path.join(RUNDIR, os.environ.get("RG_NDX", "index.ndx"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def gmx(self, args, stdin):
        p = subprocess.run(["gmx"] + args, input=stdin, text=True, capture_output=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return p.stdout + p.stderr

    def test_numbering_matches_gromacs_listing(self):
        """The zero-based number we compute from the file == the number GROMACS prints for that group."""
        out = self.gmx(["make_ndx", "-f", self.tpr, "-n", self.ndx, "-o", os.path.join(self.tmp, "x.ndx")], "q\n")
        listed = {int(m.group(1)): (m.group(2).strip(), int(m.group(3)))
                  for m in re.finditer(r"^\s*(\d+)\s+(.+?)\s*:\s+(\d+)\s+atoms", out, re.M)}
        groups = read_index_groups(self.ndx)
        self.assertTrue(listed, "could not parse the make_ndx group listing")
        for name, atoms in groups.items():
            if not atoms or list(groups).count(name) != 1:
                continue
            ref = resolve_group_ref(name, self.ndx)
            self.assertIn(ref.number, listed)
            self.assertEqual(listed[ref.number][0].lower(), name.lower(), (ref, listed[ref.number]))
            self.assertEqual(listed[ref.number][1], len(atoms), name)

    def test_rmsf_number_and_name_select_the_same_atoms(self):
        by_number = run_rmsf(self.tpr, self.traj, index_file=self.ndx, selection="C-alpha", e=1.0,
                             output_prefix=os.path.join(self.tmp, "num"))
        ref = by_number.metadata["group_resolution"]["selection"]
        self.assertEqual(ref["sent_as"], "number")
        self.gmx(["rmsf", "-s", self.tpr, "-f", self.traj, "-n", self.ndx, "-o", os.path.join(self.tmp, "name.xvg"),
                  "-res", "-fit", "-e", "1000.0"], "C-alpha\n")
        from analysis_backend.analysis_utils import parse_xvg
        by_name = parse_xvg(os.path.join(self.tmp, "name.xvg"))
        np.testing.assert_allclose(by_number.data["RMSF"].to_numpy(), by_name.iloc[:, 1].to_numpy(), rtol=1e-5)

    def test_gyrate_group_number_equals_group_name(self):
        if VINFO["major_year"] < 2024:
            self.skipTest("selection-based gyrate needs GROMACS 2024+")
        num = run_gyrate(self.tpr, self.traj, index_file=self.ndx, group="Protein", e=1.0,
                         output_prefix=os.path.join(self.tmp, "n"))
        self.assertRegex(num.metadata["selection"], r"^group \d+$")
        self.gmx(["gyrate", "-s", self.tpr, "-f", self.traj, "-n", self.ndx, "-o", os.path.join(self.tmp, "m.xvg"),
                  "-sel", 'group "Protein"', "-mode", "mass", "-rmpbc", "-e", "1000.0"], "")
        from analysis_backend.analysis_utils import parse_xvg
        by_name = parse_xvg(os.path.join(self.tmp, "m.xvg"))
        np.testing.assert_allclose(num.data["Rg"].to_numpy(), by_name.iloc[:, 1].to_numpy(), rtol=1e-5)

    def test_unknown_group_is_rejected_before_gmx_runs(self):
        with self.assertRaises(ValueError):
            run_rmsf(self.tpr, self.traj, index_file=self.ndx, selection="NoSuchGroup",
                     output_prefix=os.path.join(self.tmp, "bad"))


if __name__ == "__main__":
    unittest.main()
