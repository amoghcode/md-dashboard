"""Shared option handling for the gmx rms / rmsf / gyrate wrappers.

Additive module: nothing in analysis_utils.py is changed. It holds the pieces
that the three analyses must agree on - time handling (ns in, per-command
units out), index-group resolution, enum validation, and output sanity checks.
"""
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


NS_TO_PS = Decimal(1000)

# Canonical spellings of GROMACS' automatically generated groups. By policy of
# this wrapper they are accepted ONLY when no index file is supplied; with an
# index file, only groups defined in that file are accepted.
STANDARD_GROUP_NAMES = [
    "System", "Protein", "Protein-H", "C-alpha", "Backbone", "MainChain",
    "MainChain+Cb", "MainChain+H", "SideChain", "SideChain-H", "Prot-Masses",
    "non-Protein", "Other", "Water", "SOL", "OW", "CL", "NA", "Ions",
]
_STANDARD_BY_LOWER = {g.lower(): g for g in STANDARD_GROUP_NAMES}

# Groups that can never be a ligand in a "Protein + ligand" selection.
NON_LIGAND_GROUPS = {g.lower() for g in STANDARD_GROUP_NAMES} | {
    "water_and_ions", "non-water", "ion", "protein_and_ions"
}

# Documented defaults (GROMACS 2026 manual) used by the TUI to pre-fill forms.
ANALYSIS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "rmsd": {"fit_group": "Backbone", "calc_group": "Backbone", "fit_mode": "rot+trans",
             "mass": True, "pbc": True, "reference": "tpr structure (t = start of run)"},
    "rmsf": {"selection": "C-alpha", "res": True, "fit": True},
    "gyrate": {"group": "Protein", "weighting": "mass", "components": ("x", "y", "z"), "pbc": True},
}


# ----------------------------------------------------------------- generic ---
def require_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    raise ValueError(f"Option '{name}' must be True or False, got {value!r}.")


def validate_choice(value: Any, allowed: Sequence[str], name: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ValueError(f"Invalid {name} {value!r}. Supported values: {list(allowed)}.")
    return value.strip().lower()


# -------------------------------------------------------------------- time ---
@dataclass(frozen=True)
class TimeWindow:
    """Time selection in ns as supplied by the user (None = not set)."""
    begin_ns: float
    end_ns: Optional[float]
    step_ns: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {"begin": self.begin_ns, "end": self.end_ns, "step": self.step_ns, "unit": "ns"}


def _finite_number(x: Any, name: str) -> float:
    is_num = isinstance(x, (int, float)) or type(x).__module__ == "numpy"
    if isinstance(x, bool) or not is_num:
        raise ValueError(f"Time option '{name}' must be a number (ns), got {x!r}.")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"Time option '{name}' must be finite, got {x}.")
    return x


def normalize_time_window(b: Any = 0.0, e: Any = -1.0, dt: Any = None) -> TimeWindow:
    """Validate b/e/dt (ns). e None or negative and dt None mean 'not set'.

    e == 0 is rejected: GROMACS treats -e 0 as 'end of trajectory', so a literal
    end time of 0 ns cannot be expressed and must not be silently reinterpreted.
    """
    b = _finite_number(0.0 if b is None else b, "b")
    if b < 0:
        raise ValueError(f"Start time b must be >= 0 ns, got {b}.")
    end: Optional[float] = None
    if e is not None:
        e = _finite_number(e, "e")
        if e == 0.0:
            raise ValueError("End time e=0 is GROMACS' 'not set' value. Use e=None (or a negative "
                             "number) for end-of-trajectory, or a positive end time in ns.")
        if e > 0:
            end = e
    if end is not None and end <= b:
        raise ValueError(f"End time e={end} ns must be greater than start time b={b} ns.")
    step: Optional[float] = None
    if dt is not None:
        d = _finite_number(dt, "dt")
        if d < 0:
            raise ValueError(f"Sampling interval dt must be positive (ns), got {d}.")
        if d > 0:
            step = d
    if step is not None and end is not None and step > end - b:
        raise ValueError(f"Sampling interval dt={step} ns exceeds the selected window "
                         f"({end - b} ns); at most one frame could be used.")
    return TimeWindow(b, end, step)


def fmt_time(value_ns: float, unit: str) -> str:
    """Format a ns value for GROMACS in `unit` ('ns' or 'ps') using exact decimal
    arithmetic (1.1 ns -> '1100.0', not '1100.0000000000002')."""
    d = Decimal(str(float(value_ns)))
    if unit == "ps":
        d = d * NS_TO_PS
    elif unit != "ns":
        raise ValueError(f"Unsupported time unit for GROMACS: {unit}")
    s = format(d, "f")
    return s if "." in s else s + ".0"


def time_args(window: TimeWindow, unit: str) -> List[str]:
    """-b/-e/-dt flags with values expressed in `unit` (what the command reads)."""
    args: List[str] = []
    if window.begin_ns > 0:
        args += ["-b", fmt_time(window.begin_ns, unit)]
    if window.end_ns is not None:
        args += ["-e", fmt_time(window.end_ns, unit)]
    if window.step_ns is not None:
        args += ["-dt", fmt_time(window.step_ns, unit)]
    return args


# ------------------------------------------------------------------ groups ---
def _check_token(name: Any, what: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{what} cannot be empty.")
    name = name.strip()
    if any(c in name for c in '\r\n"\x00'):
        raise ValueError(f"{what} {name!r} contains characters (newline or quote) that are not allowed.")
    return name


def _parse_index(index_file: str):
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    names: List[str] = []
    current: Optional[str] = None
    with open(index_file, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith((";", "#")):
                continue
            m = re.fullmatch(r"\[\s*(.*?)\s*\]", line)
            if m:
                current = m.group(1)
                names.append(current)
                groups.setdefault(current, [])
                continue
            if current is None:
                raise ValueError(f"Malformed index file {index_file}: atoms before first [ group ] header.")
            try:
                groups[current].extend(int(t) for t in line.split())
            except ValueError:
                raise ValueError(f"Malformed index file {index_file}: non-integer atom entry in group "
                                 f"'{current}': {line!r}")
    dups = {n.lower() for n in names if names.count(n) > 1}
    return groups, dups


def read_index_groups(index_file: str) -> "OrderedDict[str, List[int]]":
    """Parse a .ndx file into {group name: [atom numbers]} in file order
    (file order == the numbering GROMACS shows in its prompts)."""
    return _parse_index(index_file)[0]


def _index_lookup(index_file: str):
    return _parse_index(index_file)


def resolve_index_group(name: str, index_file: str, what: str = "Group") -> str:
    """Strict: when -n is passed, tools using the classic group prompt only offer
    groups that are IN that file. Returns the file's own spelling."""
    name = _check_token(name, what)
    groups, dups = _index_lookup(index_file)
    matches = [g for g in groups if g.lower() == name.lower()]
    if not matches:
        raise ValueError(f"{what} '{name}' is not defined in index file {index_file}. "
                         f"Available groups: {list(groups)}.")
    if name.lower() in dups:
        raise ValueError(f"{what} '{name}' is defined more than once in {index_file}; ambiguous.")
    if not groups[matches[0]]:
        raise ValueError(f"{what} '{matches[0]}' in {index_file} contains no atoms.")
    return matches[0]


def resolve_standard_group(name: str, what: str = "Group") -> str:
    """No index file: only GROMACS' automatically generated groups are valid.
    Returns the canonical GROMACS spelling."""
    name = _check_token(name, what)
    canon = _STANDARD_BY_LOWER.get(name.lower())
    if canon is None:
        raise ValueError(f"{what} '{name}' is not an automatically generated GROMACS group and no index "
                         f"file was supplied. Valid automatic groups: {STANDARD_GROUP_NAMES}. "
                         "Custom groups (e.g. PROA, a ligand) require an index file defining them.")
    return canon


def resolve_group(name: str, index_file: Optional[str], what: str = "Group") -> str:
    """Single rule for every analysis: index file given -> only groups defined in
    that file; no index file -> only automatic GROMACS groups."""
    if index_file:
        return resolve_index_group(name, index_file, what)
    return resolve_standard_group(name, what)


@dataclass(frozen=True)
class GroupRef:
    """A validated atom-group reference and how it is handed to GROMACS.

    Verified against the GROMACS manual: classic group prompts (gmx rms, rmsf, gyrate before 2024) accept
    a group NUMBER or NAME; selection-based tools (gmx gyrate 2024+) accept 'group NR' (zero-based) or
    'group NAME' where NAME may match part of a name. When an index file is given the group comes from
    that file, so its zero-based position is unambiguous and is what gets sent. Without an index file the
    automatic groups depend on the .tpr contents, so the canonical name is sent.
    """
    name: str
    number: Optional[int]       # zero-based position in the index file; None for automatic groups
    source: str                 # "index_file" | "default_groups"

    @property
    def stdin_token(self) -> str:
        return str(self.number) if self.number is not None else self.name

    @property
    def selection_expr(self) -> str:
        return f"group {self.number}" if self.number is not None else f'group "{self.name}"'

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "index_number": self.number, "source": self.source,
                "sent_as": "number" if self.number is not None else "name"}


def _index_group_numbers(index_file: str) -> List[str]:
    """Group names in file order, duplicates included: this is GROMACS' zero-based group numbering."""
    _parse_index(index_file)  # validates the file (raises on malformed content)
    names: List[str] = []
    with open(index_file, "r") as fh:
        for raw in fh:
            m = re.fullmatch(r"\[\s*(.*?)\s*\]", raw.strip())
            if m and not raw.strip().startswith((";", "#")):
                names.append(m.group(1))
    return names


def resolve_group_ref(name: str, index_file: Optional[str], what: str = "Group") -> GroupRef:
    """Validate `name` (same rules as resolve_group) and return how to pass it to GROMACS."""
    canon = resolve_group(name, index_file, what)
    if not index_file:
        return GroupRef(canon, None, "default_groups")
    order = _index_group_numbers(index_file)
    hits = [i for i, n in enumerate(order) if n == canon]
    if len(hits) != 1:   # resolve_group already rejects duplicates; this guards the numbering itself
        raise ValueError(f"{what} '{canon}' does not map to exactly one group number in {index_file}.")
    return GroupRef(canon, hits[0], "index_file")


def resolve_selection_group(name: str, index_file: Optional[str], what: str = "Group") -> str:
    """Same rule as resolve_group (kept as a separate name for API compatibility)."""
    return resolve_group(name, index_file, what)


def check_ligand_group(ligand: str, base: str, index_file: Optional[str]) -> List[str]:
    """Validate a ligand group for a Protein + ligand selection. Returns warnings."""
    if not index_file:
        raise ValueError("A ligand group is custom, so an index file (-n) defining it is required.")
    ligand = resolve_index_group(ligand, index_file, "Ligand group")
    if ligand.lower() in NON_LIGAND_GROUPS:
        raise ValueError(f"'{ligand}' is a standard protein/solvent/ion group, not a ligand group.")
    if ligand.lower() == base.lower():
        raise ValueError("Ligand group and protein group are the same group.")
    warns: List[str] = []
    groups, _ = _index_lookup(index_file)
    base_hit = [g for g in groups if g.lower() == base.lower()]
    if base_hit:
        overlap = set(groups[base_hit[0]]) & set(groups[ligand])
        if overlap:
            raise ValueError(f"Ligand group '{ligand}' overlaps group '{base}' ({len(overlap)} shared atoms); "
                             "not a valid protein + ligand selection.")
    else:
        warns.append(f"Protein group '{base}' is an automatic group, so overlap with ligand group "
                     f"'{ligand}' could not be checked against the index file.")
    return warns


# ------------------------------------------------------------- xvg / output ---
def read_xvg_legends(xvg_path: str) -> List[str]:
    legends: List[str] = []
    with open(xvg_path, "r") as fh:
        for line in fh:
            m = re.match(r'@\s*s\d+\s+legend\s+"(.*)"', line.strip())
            if m:
                legends.append(m.group(1))
    return legends


def require_finite(df: pd.DataFrame, what: str) -> None:
    if not df.shape[0]:
        raise ValueError(f"{what}: output contains no rows.")
    vals = df.to_numpy(dtype=float)
    if not math.isfinite(float(vals.sum())) or (vals != vals).any():
        raise ValueError(f"{what}: output contains NaN/inf values; refusing to report it as a result.")


def time_coverage(df: pd.DataFrame, x_unit: str, window: TimeWindow) -> Dict[str, Any]:
    """Actual time span in the output (ns) vs the request. Used to expose truncated
    reads (e.g. a torn trajectory frame) instead of hiding them."""
    x = df.iloc[:, 0].to_numpy(dtype=float)
    scale = 1.0 if x_unit == "ns" else 0.001
    first, last = float(x[0]) * scale, float(x[-1]) * scale
    cov: Dict[str, Any] = {"first": first, "last": last, "n_frames": int(len(x)), "unit": "ns",
                           "requested_end": window.end_ns, "warnings": []}
    if window.end_ns is not None and last < window.end_ns * 0.99 - 1e-9:
        cov["warnings"].append(
            f"Output ends at {last:.6g} ns but e={window.end_ns:g} ns was requested: the trajectory may "
            "end earlier, or GROMACS stopped reading early (e.g. a truncated frame).")
    if window.begin_ns > 0 and first < window.begin_ns - 1e-6:
        cov["warnings"].append(f"Output starts at {first:.6g} ns, before the requested b={window.begin_ns:g} ns.")
    return cov


def is_tpr(path: str) -> bool:
    return path.lower().endswith(".tpr")


def validate_reference_structure(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom reference structure file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Custom reference structure file is empty: {path}")
    if not path.lower().endswith((".tpr", ".gro", ".g96", ".pdb", ".brk", ".ent")):
        raise ValueError("Reference structure must be .tpr, .gro, .g96, .pdb, .brk or .ent "
                         f"(formats accepted by gmx rms -s), got: {path}")
