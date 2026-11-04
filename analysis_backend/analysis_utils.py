import os
import subprocess
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re

@dataclass
class AnalysisResult:
    analysis_name: str
    output_xvg: str
    output_csv: str
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"

# Release version token of the year-based scheme (2016 onwards): YYYY.N[.P] plus an optional build
# suffix, e.g. 2026.2, 2024.1-rc1, 2023.3-Ubuntu_20230812, 2023.3-beta-20230812-abcdef.
_VERSION_TOKEN = re.compile(r"^(?P<year>\d{4})\.(?P<minor>\d+)(?:\.\d+)?(?P<suffix>[-+_~].*)?$")
_VERSION_LINE = re.compile(r"^\s*GROMACS version\s*:\s*(?P<token>\S+)", re.IGNORECASE)
_BANNER_LINE = re.compile(r":-\)\s*GROMACS\s*-\s*[^,]+,\s*(?P<token>\S+)\s*\(-:", re.IGNORECASE)
_MIN_YEAR = 2016  # first year-based release; older releases (4.x, 5.x) are reported as unknown


def _parse_gromacs_version(text: str) -> Optional[Dict[str, Any]]:
    """Extract the release from `gmx --version` text, or None if it cannot be identified.

    Only the version token of the 'GROMACS version:' line (or the '-:) GROMACS - gmx, X (-:' banner)
    is read, and it must start with YYYY.N. Dates in build suffixes, install paths or other lines are
    never interpreted as a version.
    """
    for pattern in (_VERSION_LINE, _BANNER_LINE):
        for line in text.splitlines():
            m = pattern.search(line)
            if not m:
                continue
            tok = _VERSION_TOKEN.match(m.group("token"))
            if tok and int(tok.group("year")) >= _MIN_YEAR:
                return {
                    "version_string": line.strip(),
                    "major_year": int(tok.group("year")),
                    "release": f"{tok.group('year')}.{tok.group('minor')}",
                    "build_suffix": (tok.group("suffix") or "").lstrip("-+_~") or None,
                }
    return None


def get_gromacs_version_info() -> Dict[str, Any]:
    """
    Detects the installed GROMACS release via `gmx --version`.

    Returns {"version_string", "major_year", "available", "release", "build_suffix"}:
      * gmx runs and the release is parsed -> available=True, major_year=<release year>
        (e.g. '2023.3-Ubuntu_20230812' -> 2023, '2024.1-rc1' -> 2024, '2026.2' -> 2026).
      * gmx runs but the output has no recognisable release -> available=True,
        version_string='Unknown', major_year=None. A version is never inferred from a build date
        and an unknown version is never assumed to be 2024.
      * gmx is missing, fails, or times out -> available=False, major_year=None.
    """
    try:
        result = subprocess.run(
            __import__("execution").gmx_command() + [ "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=5
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {"version_string": "GROMACS Not Available", "major_year": None, "available": False,
                "release": None, "build_suffix": None}
    text = "\n".join(s for s in (result.stdout, result.stderr) if isinstance(s, str))
    parsed = _parse_gromacs_version(text)
    if parsed is None:
        return {"version_string": "Unknown", "major_year": None, "available": True,
                "release": None, "build_suffix": None}
    parsed["available"] = True
    return parsed


def validate_inputs(tpr_file: str, traj_file: str, index_file: Optional[str] = None) -> None:
    """Validates required files, existence, readability, non-emptiness, and format compatibility."""
    for fpath, name in [(tpr_file, "Topology (.tpr)"), (traj_file, "Trajectory (.xtc/.trr/.gro)")]:
        if not fpath or not os.path.exists(fpath):
            raise FileNotFoundError(f"Required {name} file not found: {fpath}")
        if not os.access(fpath, os.R_OK):
            raise PermissionError(f"{name} file is not readable: {fpath}")
        if os.path.getsize(fpath) == 0:
            raise ValueError(f"{name} file is empty: {fpath}")
            
    if index_file:
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if os.path.getsize(index_file) == 0:
            raise ValueError(f"Index file is empty: {index_file}")

    if not tpr_file.endswith('.tpr'):
        raise ValueError(f"Topology file must be a .tpr file, got: {tpr_file}")
    if not any(traj_file.endswith(ext) for ext in ['.xtc', '.trr', '.gro']):
        raise ValueError(f"Trajectory file must be .xtc, .trr, or .gro, got: {traj_file}")

def validate_group_selection(group_name: str, index_file: Optional[str] = None) -> str:
    """Validates and normalizes group selections against index file or default GROMACS groups."""
    if not group_name or not group_name.strip():
        raise ValueError("Group selection cannot be empty.")
    
    clean_name = group_name.strip()
    standard_groups = [
        "System", "Protein", "Protein-H", "C-alpha", "Backbone",
        "MainChain", "MainChain+Cb", "MainChain+H", "SideChain", "SideChain-H",
        "Prot-Masses", "Non-Protein", "Water", "SOL", "OW", "CL", "NA", "Ions"
    ]
    
    if index_file and os.path.exists(index_file):
        found_groups = []
        with open(index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    found_groups.append(line[1:-1].strip())
        if clean_name not in found_groups and not any(clean_name.lower() == g.lower() for g in standard_groups):
            raise ValueError(
                f"Group '{clean_name}' was not found in index file {index_file} (available groups: {found_groups}) "
                f"nor in standard GROMACS groups ({standard_groups})."
            )
    else:
        if not any(clean_name.lower() == g.lower() for g in standard_groups):
            raise ValueError(
                f"Custom group '{clean_name}' requires an index file (-n). "
                f"Without an index file, selection must be one of standard GROMACS groups: {standard_groups}."
            )
            
    return clean_name

def remove_stale_outputs(xvg_path: str, csv_path: str) -> None:
    """Removes previous output files before execution to prevent reading stale or corrupted results."""
    for p in [xvg_path, csv_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

def run_gromacs_command(cmd: List[str], log_prefix: str = "gmx", stdin_input: Optional[str] = None) -> str:
    """Executes a GROMACS command via subprocess with error and timeout handling. Never returns fake results."""
    cmd_str = " ".join(cmd)
    print(f"[{log_prefix.upper()}] Executing: {cmd_str}")
    try:
        result = subprocess.run(
            cmd,
            input=stdin_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=300
        )
        return result.stdout
    except FileNotFoundError:
        error_msg = f"GROMACS binary 'gmx' not found on PATH. Execution FAILED for: {cmd_str}"
        print(f"[{log_prefix.upper()}] ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)
    except subprocess.TimeoutExpired:
        error_msg = f"GROMACS command timed out (>300s): {cmd_str}"
        print(f"[{log_prefix.upper()}] ERROR: {error_msg}")
        raise TimeoutError(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"GROMACS execution FAILED ({cmd_str}):\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        print(f"[{log_prefix.upper()}] ERROR: {error_msg}")
        raise RuntimeError(error_msg)

def parse_xvg(xvg_path: str) -> pd.DataFrame:
    """Parses GROMACS .xvg file into DataFrame. Raises ValueError if missing, empty, or malformed."""
    if not os.path.exists(xvg_path):
        raise FileNotFoundError(f"XVG output file not found: {xvg_path}")
    if os.path.getsize(xvg_path) == 0:
        raise ValueError(f"XVG output file is empty: {xvg_path}")
        
    data_rows = []
    x_label = "Time (ns)"
    y_labels = []
    
    with open(xvg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                if 'xaxis  label' in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        x_label = parts[1]
                elif 'yaxis  label' in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        y_labels.append(parts[1])
                continue
            try:
                parts = [float(val) for val in line.split() if val]
                if parts:
                    data_rows.append(parts)
            except ValueError:
                continue
                
    if not data_rows:
        raise ValueError(f"No valid numerical data parsed from XVG file: {xvg_path}")
        
    df = pd.DataFrame(data_rows)
    if len(y_labels) >= df.shape[1] - 1:
        cols = [x_label] + y_labels[:df.shape[1]-1]
    else:
        cols = [x_label] + [f"Value_{i}" for i in range(1, df.shape[1])]
    
    if len(cols) == df.shape[1]:
        df.columns = cols
        
    return df
