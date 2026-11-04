import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from analysis_backend.analysis_utils import validate_inputs, validate_group_selection, parse_xvg, get_gromacs_version_info
from analysis_backend.plotting import generate_plotly_html, launch_xmgrace
from analysis_backend.rmsd import run_rmsd
from analysis_backend.rmsf import run_rmsf
from analysis_backend.radius_of_gyration import run_gyrate

class TestMDAnalysisBackend(unittest.TestCase):
    
    def setUp(self):
        self.dummy_tpr = "dummy.tpr"
        self.dummy_traj = "dummy.xtc"
        with open(self.dummy_tpr, "w") as f:
            f.write("fake tpr content")
        with open(self.dummy_traj, "w") as f:
            f.write("fake xtc content")
            
    def tearDown(self):
        for fname in [self.dummy_tpr, self.dummy_traj, "rmsd.xvg", "rmsd.csv", "rmsf.xvg", "rmsf.csv", "radius_of_gyration.xvg", "radius_of_gyration.csv", "analysis_dashboard.html", "dummy.txt"]:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except OSError:
                    pass

    def test_version_parsing_with_suffixes(self):
        with patch("subprocess.run") as mock_run:
            # Test version string with build suffix and beta/rc tags
            mock_run.return_value = MagicMock(stdout="GROMACS version:    2023.3-beta-20230812-abcdef\nPrecision: double\n")
            vinfo = get_gromacs_version_info()
            self.assertEqual(vinfo["major_year"], 2023)
            self.assertTrue(vinfo["available"])

            mock_run.return_value = MagicMock(stdout="GROMACS version:    2024.1-rc1\n")
            vinfo2 = get_gromacs_version_info()
            self.assertEqual(vinfo2["major_year"], 2024)

            # Unparseable version
            mock_run.return_value = MagicMock(stdout="GROMACS version:    unknown-dev\n")
            vinfo3 = get_gromacs_version_info()
            self.assertIsNone(vinfo3["major_year"])

    def test_time_conversion_rg(self):
        with patch("analysis_backend.radius_of_gyration.run_gromacs_command") as mock_gmx, \
             patch("analysis_backend.radius_of_gyration.get_gromacs_version_info") as mock_vinfo:
            mock_vinfo.return_value = {"version_string": "GROMACS 2024.1", "major_year": 2024, "available": True}
            
            def write_xvg(*args, **kwargs):
                with open("radius_of_gyration.xvg", "w") as f:
                    f.write('@    title "Rg"\n0 2.5 2.0 1.5 1.2\n1000 2.4 2.0 1.5 1.2\n')
                return ""
            mock_gmx.side_effect = write_xvg
            
            # Pass b=1.0 ns, e=10.0 ns, dt=0.1 ns -> should be converted to ps (1000.0, 10000.0, 100.0)
            res = run_gyrate(self.dummy_tpr, self.dummy_traj, b=1.0, e=10.0, dt=0.1)
            self.assertEqual(res.analysis_name, "Radius of Gyration")
            
            # Verify command arguments received multiplied by 1000
            called_args = mock_gmx.call_args[0][0]
            self.assertIn("-b", called_args)
            self.assertIn("1000.0", called_args)
            self.assertIn("-e", called_args)
            self.assertIn("10000.0", called_args)

    def test_input_validation_and_invalid_inputs(self):
        validate_inputs(self.dummy_tpr, self.dummy_traj)
        with self.assertRaises(FileNotFoundError):
            validate_inputs("nonexistent.tpr", self.dummy_traj)
        
        with open("dummy.txt", "w") as f:
            f.write("text")
        with self.assertRaises(ValueError):
            validate_inputs("dummy.txt", self.dummy_traj)

    def test_group_validation(self):
        grp = validate_group_selection("Protein")
        self.assertEqual(grp, "Protein")
        with self.assertRaises(ValueError):
            validate_group_selection("")

    def test_custom_reference_not_found(self):
        with self.assertRaises(FileNotFoundError):
            run_rmsd(self.dummy_tpr, self.dummy_traj, ref_structure="nonexistent_ref.pdb")

    @patch("analysis_backend.rmsd.run_gromacs_command")
    def test_rmsd_execution_and_defaults(self, mock_gmx):
        def write_xvg(*args, **kwargs):
            with open("rmsd.xvg", "w") as f:
                f.write('@    title "RMSD"\n@    xaxis  label "Time (ns)"\n@    yaxis  label "RMSD (nm)"\n0.0 0.15\n1.0 0.18\n')
            return ""
            
        mock_gmx.side_effect = write_xvg
        res = run_rmsd(self.dummy_tpr, self.dummy_traj)
        self.assertEqual(res.analysis_name, "RMSD")
        self.assertTrue(os.path.exists(res.output_csv))
        self.assertTrue(os.path.exists(res.output_xvg))

    @patch("analysis_backend.rmsf.get_gromacs_version_info")
    @patch("analysis_backend.rmsf.subprocess.Popen")
    def test_rmsf_execution_and_defaults(self, mock_popen, mock_vinfo):
        mock_vinfo.return_value = {"version_string": "GROMACS 2024.1", "major_year": 2024, "available": True}
        
        def communicate_side_effect(*args, **kwargs):
            with open("rmsf.xvg", "w") as f:
                f.write('@    title "RMSF"\n@    xaxis  label "Residue"\n@    yaxis  label "RMSF (nm)"\n1 0.12\n2 0.15\n3 0.20\n4 0.18\n')
            return ("STDOUT", "")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.side_effect = communicate_side_effect
        mock_popen.return_value = mock_proc
            
        res = run_rmsf(self.dummy_tpr, self.dummy_traj, selection="C-alpha")
        self.assertEqual(res.analysis_name, "RMSF")
        self.assertTrue(os.path.exists(res.output_csv))

    def test_command_failure_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("gmx not found")):
            with self.assertRaises(FileNotFoundError):
                run_rmsd(self.dummy_tpr, self.dummy_traj)

if __name__ == "__main__":
    unittest.main()
