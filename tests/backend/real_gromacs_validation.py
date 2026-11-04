"""REAL-GROMACS validation (not mocked). Skipped unless `gmx` is on PATH and WT_RUNDIR is set.

Usage:
    WT_RUNDIR=/path/to/run_directory python -m unittest tests.real_gromacs_validation -v
Optional: RG_TPR (default prod1.tpr), RG_TRAJ (default prod1.part0017.trr), RG_NDX (default index.ndx).

Use a trajectory whose atom count matches the .tpr. In the reference package that is the coarse
.trr tier (78,939 atoms, 100 ps spacing) with prod1.tpr - NOT the 62,055-atom fine .xtc.
Outputs go to a temp dir; nothing is written into run_directory.
"""
import os
import shutil
import tempfile
import unittest

from analysis_backend.analysis_utils import get_gromacs_version_info
from analysis_backend.rmsd import run_rmsd
from analysis_backend.rmsf import run_rmsf
from analysis_backend.radius_of_gyration import run_gyrate

RUNDIR = os.environ.get("WT_RUNDIR")
VINFO = get_gromacs_version_info()


@unittest.skipUnless(VINFO["available"] and RUNDIR, "needs real gmx on PATH and WT_RUNDIR")
class RealGromacs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.prefix = os.path.join(self.tmp, "out")
        self.tpr = os.path.join(RUNDIR, os.environ.get("RG_TPR", "prod1.tpr"))
        self.traj = os.path.join(RUNDIR, os.environ.get("RG_TRAJ", "prod1.part0017.trr"))
        self.ndx = os.path.join(RUNDIR, os.environ.get("RG_NDX", "index.ndx"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rmsd_time_unit_and_window(self):
        # Proves -tu ns also sets the unit of -b/-e/-dt: a 470-480 ns window must start near 470 in ns.
        r = run_rmsd(self.tpr, self.traj, index_file=self.ndx, b=470.0, e=480.0, dt=0.5, output_prefix=self.prefix)
        t = r.data.iloc[:, 0]
        self.assertGreaterEqual(t.iloc[0], 469.9)
        self.assertLessEqual(t.iloc[-1], 480.1)

    def test_rmsd_fit_none_prompt_handling(self):
        # Proves the stdin answer order is right for -fit none (RMSD of unfitted group must differ from fitted).
        fitted = run_rmsd(self.tpr, self.traj, index_file=self.ndx, e=470.0, output_prefix=self.prefix + "_f")
        nofit = run_rmsd(self.tpr, self.traj, index_file=self.ndx, e=470.0, fit_mode="none",
                         output_prefix=self.prefix + "_n")
        self.assertEqual(len(fitted.data), len(nofit.data))
        self.assertNotAlmostEqual(fitted.statistics["mean"], nofit.statistics["mean"], places=4)

    def test_rmsf_ps_conversion(self):
        # Window of 10 ns given in ns must equal the same window in a direct gmx call in ps: here we only
        # check it ran on a non-empty subset and per-residue output is produced.
        r = run_rmsf(self.tpr, self.traj, index_file=self.ndx, b=470.0, e=480.0, output_prefix=self.prefix)
        self.assertGreater(len(r.data), 10)
        self.assertTrue((r.data["RMSF"] > 0).all())

    def test_gyrate_defaults_and_legends(self):
        r = run_gyrate(self.tpr, self.traj, index_file=self.ndx, b=470.0, e=480.0, output_prefix=self.prefix)
        self.assertGreaterEqual(r.data["Time (ps)"].iloc[0], 469_900)
        self.assertFalse(any("No legends" in w for w in r.metadata["warnings"]),
                         "xvg legends absent: column labels fell back to positional")
        self.assertTrue(r.data["Rg"].between(1, 10).all())  # nm

    def test_gyrate_geometry_differs_from_mass(self):
        if VINFO["major_year"] < 2024:
            self.skipTest("geometry mode needs GROMACS 2024+")
        m = run_gyrate(self.tpr, self.traj, index_file=self.ndx, e=50.0, output_prefix=self.prefix + "_m")
        g = run_gyrate(self.tpr, self.traj, index_file=self.ndx, e=50.0, weighting="geometry",
                       output_prefix=self.prefix + "_g")
        self.assertNotAlmostEqual(m.statistics["mean"], g.statistics["mean"], places=5)


if __name__ == "__main__":
    unittest.main()
