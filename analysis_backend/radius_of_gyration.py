import os
import re
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Sequence
from .analysis_utils import AnalysisResult, validate_inputs, validate_group_selection, run_gromacs_command, parse_xvg, get_gromacs_version_info, remove_stale_outputs
from .statistics import compute_basic_summary, infer_timestep, statistical_reality_check, MIN_SAMPLES
from .plotting import get_xmgrace_command, launch_xmgrace
from .gmx_options import (
    require_bool, validate_choice, normalize_time_window, time_args, resolve_group,
    resolve_selection_group, resolve_group_ref, check_ligand_group, read_xvg_legends, require_finite, time_coverage,
)

# gmx gyrate -mode (GROMACS 2024+): mass, charge, geometry
GYRATE_WEIGHTINGS = ("mass", "charge", "geometry")
GYRATE_COMPONENTS = ("x", "y", "z")
_EXPECTED_LEGENDS = ["rg", "rgx", "rgy", "rgz"]


def _norm_legend(s: str) -> str:
    """Canonicalise known plain and Grace-subscript series labels only."""
    label = s.strip().lower()
    if label == "rg":
        return "rg"
    plain = re.fullmatch(r"rg_?([xyz])|rg\(([xyz])\)", label)
    if plain:
        return "rg" + next(group for group in plain.groups() if group)
    # GROMACS XVG uses /s ... /N (or Grace backslash equivalents)
    # to start a subscript and restore normal text.
    formatted = re.fullmatch(r"rg[/\\]s([xyz])[/\\]n", label)
    if formatted:
        return "rg" + formatted.group(1)
    return label



def _total_rg_column(legends: Sequence[str]) -> Optional[int]:
    """0-based index (among the data columns after time) of the TOTAL Rg series, identified from the
    XVG legends. None if there is no legend or it is not unambiguous - never guessed by position."""
    hits = [i for i, l in enumerate(legends) if _norm_legend(l) == "rg"]
    return hits[0] if len(hits) == 1 else None


def _xvg_x_label(xvg_path: str) -> Optional[str]:
    """Explicit x-axis label of the xvg, or None (parse_xvg substitutes a default, so it cannot tell)."""
    with open(xvg_path, "r") as fh:
        for line in fh:
            m = re.match(r'@\s*xaxis\s+label\s+"(.*)"', line.strip())
            if m:
                return m.group(1)
    return None


def _skip(code: str, reason: str) -> Dict[str, Any]:
    return {"status": "not_run", "reason_code": code, "reason": reason}


def _reality_check_for_rg(times_ps: np.ndarray, rg: np.ndarray, rg_legend_index: Optional[int],
                          legend: Optional[str], x_label: Optional[str]) -> Dict[str, Any]:
    """Statistical Reality Check on the ACTUAL total-Rg series and timestamps of a gmx gyrate run.

    Times in the xvg are ps (gmx gyrate is run without -tu); they are converted to ns, the unit the
    check reports. Never raises: the Rg calculation already succeeded, so a skip or failure is
    recorded with its reason instead of discarding or inventing a result.
    """
    if rg_legend_index is None:
        return _skip("no_total_rg_legend", "The total Rg column could not be identified from the .xvg legends "
                     "(no unambiguous 'Rg' legend), and columns are not assumed by position.")
    if x_label is not None and "ps" not in x_label.lower():
        return _skip("time_unit_unverified", f"The .xvg time axis is labelled '{x_label}', not ps; timestamps "
                     "cannot be converted to ns reliably.")
    if len(rg) < MIN_SAMPLES:
        return _skip("insufficient_data", f"Only {len(rg)} frames (< {MIN_SAMPLES}); too few for autocorrelation, "
                     "equilibration and block-averaging diagnostics.")
    times_ns = np.asarray(times_ps, dtype=float) / 1000.0
    ts = infer_timestep(times_ns)
    if not ts["uniform"] or not ts["dt"]:
        return _skip("irregular_sampling", f"Frames are not equally spaced ({ts['reason']}); the autocorrelation-"
                     "based diagnostics require uniform spacing, so they were not computed.")
    try:
        full = statistical_reality_check(np.asarray(rg, dtype=float), dt=ts["dt"], time_unit="ns",
                                         value_unit="nm", times=times_ns)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    eq, ess, conv = full["equilibration"], full["effective_sample_size"], full["block_averaging"]["convergence"]
    return {
        "status": full["overall"]["status"],
        "series": {"observable": "total Rg", "legend": legend, "xvg_data_column": rg_legend_index + 1,
                   "identified_by": "xvg legend"},
        "summary": {
            "overall_status": full["overall"]["status"],
            "reasons": list(full["overall"]["reasons"]),
            "candidate_equilibration": {"status": eq["status"], "method": eq["method"],
                                        "t0_frame": eq["t0_frame"], "t0_time": eq["t0_time"],
                                        "t0_elapsed": eq["t0_elapsed"], "time_origin": eq["time_origin"],
                                        "time_unit": "ns"},
            "analysis_segment": full["analysis_segment"],
            "mean": ess["mean"], "ess": ess["ess"], "tau_int_time": ess["tau_int_time"],
            "sem_naive": ess["sem_naive"], "sem_corrected": ess["sem_corrected"], "value_unit": "nm",
            "block_convergence": {"status": conv["status"], "sem_plateau": conv.get("sem_plateau"),
                                  "implied_ess": conv.get("implied_ess")},
            "warnings": list(full["warnings"]),
        },
        "full": full,
        "frame_spacing": {"dt": ts["dt"], "unit": "ns", "n_frames": int(len(rg))},
    }


def run_gyrate(
    tpr_file: str,
    traj_file: str,
    index_file: Optional[str] = None,
    group: str = "Protein",
    mass: bool = True,
    b: float = 0.0,
    e: float = -1.0,
    dt: Optional[float] = None,
    pbc: bool = True,
    output_prefix: str = "radius_of_gyration",
    ligand_group: Optional[str] = None,
    weighting: Optional[str] = None,
    components: Sequence[str] = GYRATE_COMPONENTS
) -> AnalysisResult:
    """
    Runs gmx gyrate for compactness analysis.
    Scientific Question: Is the protein becoming more or less compact?
    Default Command: gmx gyrate

    GROMACS 2024 replaced gmx gyrate by a selection-based tool (the old one is gmx gyrate-legacy).
    Both are supported and chosen from the installed version:
      >= 2024  gmx gyrate -sel '<selection>' -mode mass|charge|geometry -rmpbc|-normpbc
      <  2024  gmx gyrate, group on stdin; -q for charge weighting. Geometric weighting and
               pbc=False do not exist there and raise ValueError.

    Options:
      group         protein group (default Protein) - or an individual chain / custom group.
      ligand_group  adds a ligand: selection  group "<group>" or group "<ligand>". Needs the
                    selection syntax (>= 2024) and an index group that is not protein/solvent/ion
                    and does not overlap `group`. On older versions pass a pre-made combined group
                    as `group` instead.
      weighting     mass (default) | charge | geometry. When None it follows `mass`
                    (True -> mass, False -> geometry); an explicit value that contradicts
                    `mass` raises ValueError.
      components    which of x, y, z radii are kept in the CSV (default all). gmx gyrate always
                    writes them to the .xvg; the .xvg itself is never modified.
      b, e, dt      start / end / sampling interval in ns, converted to ps (the unit of
                    -b/-e/-dt here; -tu is left at its default, so output time is in ps).
    """
    validate_inputs(tpr_file, traj_file, index_file)
    mass = require_bool(mass, "mass")
    pbc = require_bool(pbc, "pbc")
    window = normalize_time_window(b, e, dt)

    if weighting is None:
        weighting = "mass" if mass else "geometry"
    else:
        weighting = validate_choice(weighting, GYRATE_WEIGHTINGS, "weighting")
        if weighting == "mass" and not mass:
            raise ValueError("Conflicting options: weighting='mass' but mass=False.")
    mass_weighted = weighting == "mass"

    if isinstance(components, str) or components is None:
        raise ValueError(f"components must be a sequence drawn from {list(GYRATE_COMPONENTS)}, got {components!r}.")
    comps: List[str] = []
    for c in components:
        c = validate_choice(c, GYRATE_COMPONENTS, "component")
        if c in comps:
            raise ValueError(f"Duplicate component {c!r}.")
        comps.append(c)

    vinfo = get_gromacs_version_info()
    major_year = vinfo["major_year"]
    if not vinfo.get("available", True):
        raise FileNotFoundError("GROMACS binary 'gmx' not found on PATH; cannot run gmx gyrate.")
    if major_year is None:
        raise ValueError(
            "GROMACS version could not be parsed, but gmx gyrate changed incompatibly in 2024 "
            "(selection-based vs. legacy). Refusing to guess the command line.")
    new_tool = major_year >= 2024

    warnings: List[str] = []
    if new_tool:
        base_ref = resolve_group_ref(group, index_file, "Group")
        base = base_ref.name
        # index file given -> 'group <zero-based number>' (unambiguous; NAME may match part of a name)
        selection = base_ref.selection_expr
        readable = f'group "{base}"'
        ligand_ref = None
        if ligand_group is not None:
            warnings.extend(check_ligand_group(ligand_group, base, index_file))
            ligand_ref = resolve_group_ref(ligand_group, index_file, "Ligand group")
            ligand = ligand_ref.name
            selection += f" or {ligand_ref.selection_expr}"
            readable += f' or group "{ligand}"'
            warnings.append("Protein + ligand: -rmpbc makes each molecule whole separately; GROMACS does "
                            "not keep a diffusing ligand next to the protein, so check ligand placement.")
        else:
            ligand = None
    else:
        if ligand_group is not None:
            raise ValueError(
                f"Protein + ligand selections need the selection syntax of GROMACS 2024+ "
                f"(installed: {vinfo['version_string']}). Pass an existing combined index group as `group`.")
        if weighting == "geometry":
            raise ValueError(
                f"GROMACS version {vinfo['version_string']} does not support unweighted (geometric) Rg via "
                "gmx gyrate (-mode geometry exists from 2024). Use weighting='mass' or 'charge'.")
        if not pbc:
            raise ValueError(
                f"GROMACS version {vinfo['version_string']}: legacy gmx gyrate has no option to switch off "
                "making molecules whole (-normpbc exists from 2024); pbc=False is unsupported.")
        base_ref = resolve_group_ref(group, index_file, "Group")
        base = base_ref.name
        selection = base
        readable = base
        ligand_ref = None
        ligand = None

    output_xvg = f"{output_prefix}.xvg"
    output_csv = f"{output_prefix}.csv"
    remove_stale_outputs(output_xvg, output_csv)

    cmd = __import__("execution").gmx_command() + [ "gyrate", "-s", tpr_file, "-f", traj_file, "-o", output_xvg]
    if index_file:
        cmd.extend(["-n", index_file])
    if new_tool:
        cmd.extend(["-sel", selection, "-mode", weighting])
        cmd.append("-rmpbc" if pbc else "-normpbc")
        group_input = ""  # selection given on the command line: nothing to prompt for
    else:
        if weighting == "charge":
            cmd.append("-q")
        group_input = f"{base_ref.stdin_token}\n"   # group number with an index file, else the name
    cmd.extend(time_args(window, "ps"))  # ns -> ps

    run_gromacs_command(cmd, log_prefix="gyrate", stdin_input=group_input)

    df = parse_xvg(output_xvg)
    if df.shape[1] != 5:
        raise ValueError(f"Unexpected gmx gyrate output: expected 5 columns (time, Rg, Rg_X, Rg_Y, Rg_Z), "
                         f"got {df.shape[1]}.")
    legends = read_xvg_legends(output_xvg)
    if legends:
        if [_norm_legend(l) for l in legends] != _EXPECTED_LEGENDS:
            raise ValueError(f"gmx gyrate legends {legends} do not match the expected Rg, Rg_X, Rg_Y, Rg_Z "
                             "order; refusing to label columns by position.")
    else:
        warnings.append("No legends in gmx gyrate .xvg; columns labelled by position (Rg, X, Y, Z).")
    # total Rg is located from the legends (before any renaming), never by assumed position
    rg_idx = _total_rg_column(legends) if legends else None
    rg_legend = legends[rg_idx] if rg_idx is not None else None
    x_label = _xvg_x_label(output_xvg)
    df.columns = ["Time (ps)", "Rg", "Rg_X", "Rg_Y", "Rg_Z"]
    require_finite(df, "Radius of gyration")
    raw_times_ps = df["Time (ps)"].to_numpy(dtype=float)
    raw_rg = df.iloc[:, 1 + rg_idx].to_numpy(dtype=float) if rg_idx is not None else np.array([])
    df = df[["Time (ps)", "Rg"] + [f"Rg_{c.upper()}" for c in comps]]
    df.to_csv(output_csv, index=False)

    series = df["Rg"].values
    summary = compute_basic_summary(series)
    coverage = time_coverage(df, "ps", window)
    warnings.extend(coverage.pop("warnings"))

    reality = _reality_check_for_rg(raw_times_ps, raw_rg, rg_idx, rg_legend, x_label)
    if reality["status"] in ("not_run", "error"):
        warnings.append("Statistical reality check " + reality["status"].replace("_", " ") + ": " +
                        reality.get("reason", reality.get("error", "")))
    else:
        warnings.extend(f"Statistical check: {w}" for w in reality["summary"]["warnings"])

    metadata = {
        "scientific_question": "Is the protein becoming more or less compact?",
        "analysis_type": "Radius of Gyration",
        "command": "gmx gyrate",
        "status": "SUCCESS",
        "group": base,
        "mass_weighted": mass_weighted,
        "pbc": pbc,
        "time_unit": "ps (converted from ns)",
        "tpr_file": tpr_file,
        "traj_file": traj_file,
        "gmx_version": vinfo["version_string"],
        "xmgrace_cmd": get_xmgrace_command(output_xvg),
        # --- TUI integration: selected parameters, units, provenance ---
        "index_file": index_file,
        "weighting": weighting,
        "ligand_group": ligand,
        "selection": selection,
        "selection_readable": readable,
        "group_resolution": {"group": base_ref.as_dict(),
                             "ligand_group": ligand_ref.as_dict() if ligand_ref else None},
        "gyrate_implementation": "selection-based (2024+)" if new_tool else "legacy (<2024)",
        "parameters": {
            "group": base, "ligand_group": ligand, "selection": selection, "weighting": weighting,
            "pbc": pbc, "components_in_csv": comps, "time_window": window.as_dict(),
        },
        "units": {"time": "ps", "rg": "nm", "time_window_input": "ns",
                  "time_passed_to_gmx": "ps (converted from ns)"},
        "gmx_command": list(cmd),
        "gmx_stdin_groups": [] if new_tool else [base],
        "gmx_stdin_input": [] if new_tool else [base_ref.stdin_token],
        "time_coverage": coverage,
        "warnings": warnings,
        "statistical_reality_check": reality,
    }

    return AnalysisResult(
        analysis_name="Radius of Gyration",
        output_xvg=output_xvg,
        output_csv=output_csv,
        data=df,
        metadata=metadata,
        statistics=summary
    )
