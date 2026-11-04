"""Tests for analysis_utils.get_gromacs_version_info.

All MOCKED: subprocess.run is replaced with canned `gmx --version` text. This checks parsing logic
only; it is not validation against an installed GROMACS.
"""
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from analysis_backend.analysis_utils import get_gromacs_version_info


def version_output(version_line_value, extra_before=""):
    """Text shaped like `gmx --version` (label, then build info)."""
    return (
        f"{extra_before}"
        f"GROMACS version:    {version_line_value}\n"
        "Precision:          mixed\n"
        "Memory model:       64 bit\n"
        "MPI library:        thread_mpi\n"
        "OpenMP support:     enabled (GMX_OPENMP_MAX_THREADS = 128)\n"
        "GPU support:        disabled\n"
        "Data prefix:        /opt/gromacs-2019.6\n"
        "Working dir:        /home/user/20230101\n"
        "Command line:\n  gmx --version\n"
    )


def info_for(stdout, stderr=""):
    with patch("subprocess.run") as run:
        run.return_value = MagicMock(stdout=stdout, stderr=stderr)
        return get_gromacs_version_info()


class TestVersionParsing(unittest.TestCase):
    def test_required_version_strings(self):
        cases = [
            ("2023.3-Ubuntu_20230812", 2023, "2023.3", "Ubuntu_20230812"),
            ("2024.1-rc1", 2024, "2024.1", "rc1"),
            ("2026.2", 2026, "2026.2", None),
        ]
        for raw, year, release, suffix in cases:
            info = info_for(version_output(raw))
            self.assertTrue(info["available"], raw)
            self.assertEqual(info["major_year"], year, raw)
            self.assertEqual(info["release"], release, raw)
            self.assertEqual(info["build_suffix"], suffix, raw)
            self.assertIn(raw, info["version_string"])

    def test_other_real_world_suffixes(self):
        for raw, year in [("2023.3-beta-20230812-abcdef", 2023), ("2022.5-dev-20230101-1a2b3c-dirty", 2022),
                          ("2025.0.1", 2025), ("2021.4-Ubuntu_2021.4-2_amd64", 2021),
                          ("2024.3-conda_forge", 2024), ("2019.6", 2019)]:
            self.assertEqual(info_for(version_output(raw))["major_year"], year, raw)

    def test_version_never_inferred_from_build_date(self):
        # Build date 2024-08-12 in the suffix must not turn a 2021 release into 2024.
        info = info_for(version_output("2021.4-Ubuntu_20240812"))
        self.assertEqual(info["major_year"], 2021)
        # Install path / working dir containing other years are ignored too.
        self.assertEqual(info_for(version_output("2022.1"))["major_year"], 2022)

    def test_version_line_wins_over_earlier_lines_with_years(self):
        out = version_output("2023.3", extra_before="Note: built 2025-01-01 at /srv/gromacs-2019/\n")
        self.assertEqual(info_for(out)["major_year"], 2023)

    def test_banner_form_is_understood(self):
        out = "                       :-) GROMACS - gmx, 2024.1-rc1 (-:\n\nExecutable: /usr/bin/gmx\n"
        info = info_for(out)
        self.assertEqual((info["major_year"], info["release"], info["build_suffix"]), (2024, "2024.1", "rc1"))

    def test_version_read_from_stderr_if_stdout_empty(self):
        self.assertEqual(info_for("", stderr=version_output("2025.2"))["major_year"], 2025)

    def test_malformed_output_is_unknown_not_guessed(self):
        malformed = [
            "",
            "garbage",
            "Command line:\n  gmx --version\n",
            "GROMACS version:\n",
            "GROMACS version:    unknown\n",
            "GROMACS version:    2024\n",                 # no release number
            "GROMACS version:    20230812\n",             # a date, not a release
            "GROMACS version:    v2024.1\n",              # not the documented format
            "GROMACS version:    5.1.4\n",                # pre-2016 scheme: no year to report
            "GROMACS version:    2015.3\n",               # below first year-based release
            "GROMACS version:    abc2024.1\n",
            "Build date: 20240101\nData prefix: /opt/gromacs-2024.1\nGROMACS modification: No\n",
            "GROMACS modification: No\n",
        ]
        for text in malformed:
            info = info_for(text)
            self.assertIsNone(info["major_year"], repr(text))
            self.assertEqual(info["version_string"], "Unknown", repr(text))
            self.assertTrue(info["available"], repr(text))  # gmx ran, output just unusable

    def test_non_text_output_is_unknown(self):
        with patch("subprocess.run") as run:
            run.return_value = MagicMock()  # stdout/stderr are MagicMocks, not str
            info = get_gromacs_version_info()
        self.assertIsNone(info["major_year"])
        self.assertEqual(info["version_string"], "Unknown")

    def test_missing_or_failing_gromacs_is_unavailable(self):
        for exc in [FileNotFoundError("gmx"), PermissionError("denied"),
                    subprocess.CalledProcessError(1, "gmx"), subprocess.TimeoutExpired("gmx", 5)]:
            with patch("subprocess.run", side_effect=exc):
                info = get_gromacs_version_info()
            self.assertFalse(info["available"], repr(exc))
            self.assertIsNone(info["major_year"], repr(exc))
            self.assertEqual(info["version_string"], "GROMACS Not Available")

    def test_result_keys_are_backward_compatible(self):
        for info in (info_for(version_output("2026.2")), info_for("junk")):
            for key in ("version_string", "major_year", "available"):
                self.assertIn(key, info)

    def test_command_and_timeout(self):
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(stdout=version_output("2026.2"), stderr="")
            get_gromacs_version_info()
        self.assertEqual(run.call_args[0][0], ["gmx", "--version"])
        self.assertEqual(run.call_args[1]["timeout"], 5)


if __name__ == "__main__":
    unittest.main()
