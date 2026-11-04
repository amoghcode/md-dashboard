import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from .analysis_utils import AnalysisResult, validate_inputs, validate_group_selection, run_gromacs_command, parse_xvg, get_gromacs_version_info, remove_stale_outputs
from .statistics import compute_basic_summary, infer_timestep, statistical_reality_check
from .plotting import get_xmgrace_command, launch_xmgrace
from .gmx_options import (
    require_bool, validate_choice, normalize_time_window, time_args, resolve_group, resolve_group_ref,
    validate_reference_structure, is_tpr, require_finite, time_coverage,
)

# gmx rms -fit (GROMACS 2022-2026 manual): rot+trans, translation, none
RMSD_FIT_MODES = ("rot+trans", "translation", "none")


def _reality_check_for_rmsd(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the Statistical Reality Check on the ACTUAL RMSD column of a finished gmx rms run.

    Never raises: the RMSD calculation itself already succeeded, so a failure here is recorded
    (status 'error' or 'not_run') instead of discarding or faking a result.
    """
    if df.shape[1] < 2:
        return {"status": "not_run", "reason": "RMSD output has no value column."}
    ts = infer_timestep(df.iloc[:, 0].to_numpy(dtype=float))
    if not ts["uniform"] or not ts["dt"]:
        return {"status": "not_run",
                "reason": f"Frames are not equally spaced ({ts['reason']}); the autocorrelation-based "
                          "diagnostics require uniform spacing, so they were not computed."}
    try:
        full = statistical_reality_check(df.iloc[:, 1].to_numpy(dtype=float), dt=ts["dt"],
                                         time_unit="ns", value_unit="nm",
                                         times=df.iloc[:, 0].to_numpy(dtype=float))
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    eq, ess, conv = full["equilibration"], full["effective_sample_size"], full["block_averaging"]["convergence"]
    return {
        "status": full["overall"]["status"],
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
        "frame_spacing": {"dt": ts["dt"], "unit": "ns", "n_frames": int(len(df))},
    }


def run_rmsd(
    tpr_file: str,
    traj_file: str,
    index_file: Optional[str] = None,
    fit_group: str = "Backbone",
    calc_group: str = "Backbone",
    ref_structure: Optional[str] = None,
    mass: bool = True,
    fit_mode: str = "rot+trans",
    pbc: bool = True,
    b: float = 0.0,
    e: float = -1.0,
    dt: Optional[float] = None,
    output_prefix: str = "rmsd"
) -> AnalysisResult:
    """
    Runs gmx rms for structural deviation analysis.
    Scientific Question: Has the protein's structure changed?
    Default Command: gmx rms

    Options (checked against the GROMACS 2022-2026 gmx rms manual):
      fit_group / calc_group  least-squares fit group and RMSD group. Default Backbone/Backbone.
                              With an index file, only groups defined in that file are valid.
      ref_structure           reference given to -s (default: the .tpr structure). gmx rms takes ONE -s,
                              so with a custom reference the .tpr is not read; the reference must
                              contain exactly the atoms of the trajectory (and of the index file).
      fit_mode                rot+trans | translation | none  (-fit)
      mass                    mass-weighted superposition (-mw / -nomw)
      pbc                     -pbc / -nopbc ("PBC check": molecules made whole per frame, needs bonded
                              topology, i.e. a .tpr; it does not repair jumps of whole molecules)
      b, e, dt                start / end / sampling interval in ns. gmx rms accepts -tu, which also sets
                              the unit of -b/-e/-dt, so the values are passed as ns with -tu ns and the
                              output time axis is in ns.
    """
    validate_inputs(tpr_file, traj_file, index_file)
    fit_mode = validate_choice(fit_mode, RMSD_FIT_MODES, "fit_mode")
    mass = require_bool(mass, "mass")
    pbc = require_bool(pbc, "pbc")
    window = normalize_time_window(b, e, dt)
    fit_ref = resolve_group_ref(fit_group, index_file, "Fit group")
    calc_ref = resolve_group_ref(calc_group, index_file, "RMSD group")
    fit_group, calc_group = fit_ref.name, calc_ref.name

    warnings: List[str] = []
    if ref_structure is not None:
        validate_reference_structure(ref_structure)
        struct_ref = ref_structure
    else:
        struct_ref = tpr_file
    ref_is_tpr = is_tpr(struct_ref)
    if not ref_is_tpr:
        warnings.append("Custom non-.tpr reference: atom count/order must match the trajectory and index "
                        "file exactly, and the .tpr is not read by gmx rms.")
        if mass:
            warnings.append("Mass weighting with a non-.tpr reference: GROMACS deduces masses from "
                            "atommass.dat (deprecated); verify they are correct for all atoms.")
        if pbc:
            warnings.append("pbc=True needs bonded topology to make molecules whole; a non-.tpr reference "
                            "provides none, so no whole-molecule correction may take place.")
    if fit_mode == "none":
        warnings.append("fit_mode='none': no superposition, so fit_group is not used (RMSD includes "
                        "rotation/translation).")

    output_xvg = f"{output_prefix}.xvg"
    output_csv = f"{output_prefix}.csv"
    remove_stale_outputs(output_xvg, output_csv)

    cmd = __import__("execution").gmx_command() + [ "rms", "-s", struct_ref, "-f", traj_file, "-o", output_xvg, "-tu", "ns"]
    if index_file:
        cmd.extend(["-n", index_file])
    cmd.append("-mw" if mass else "-nomw")
    cmd.extend(["-fit", fit_mode])
    cmd.append("-pbc" if pbc else "-nopbc")
    cmd.extend(time_args(window, "ns"))

    # Prompt order is fit group, then RMSD group. With -fit none the fit group
    # is irrelevant, so the RMSD group answers both prompts: correct whether or
    # not the installed version asks the fit question.
    stdin_groups = [calc_group, calc_group] if fit_mode == "none" else [fit_group, calc_group]
    # With an index file the group NUMBER is sent (unambiguous); automatic groups are sent by name.
    stdin_tokens = [calc_ref.stdin_token] * 2 if fit_mode == "none" else [fit_ref.stdin_token, calc_ref.stdin_token]
    group_input = "".join(f"{t}\n" for t in stdin_tokens)

    version = get_gromacs_version_info()
    run_gromacs_command(cmd, log_prefix="rmsd", stdin_input=group_input)

    df = parse_xvg(output_xvg)
    require_finite(df, "RMSD")
    df.to_csv(output_csv, index=False)

    val_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    series = df[val_col].values
    summary = compute_basic_summary(series)
    coverage = time_coverage(df, "ns", window)
    warnings.extend(coverage.pop("warnings"))

    reality = _reality_check_for_rmsd(df)
    if reality["status"] in ("not_run", "error"):
        warnings.append("Statistical reality check " + reality["status"].replace("_", " ") + ": " +
                        reality.get("reason", reality.get("error", "")))
    else:
        warnings.extend(f"Statistical check: {w}" for w in reality["summary"]["warnings"])

    metadata = {
        "scientific_question": "Has the protein's structure changed?",
        "analysis_type": "RMSD",
        "command": "gmx rms",
        "status": "SUCCESS",
        "fit_group": fit_group,
        "calc_group": calc_group,
        "ref_structure": struct_ref,
        "mass_weighted": mass,
        "fit_mode": fit_mode,
        "pbc": pbc,
        "time_unit": "ns",
        "tpr_file": tpr_file,
        "traj_file": traj_file,
        "gmx_version": version["version_string"],
        "xmgrace_cmd": get_xmgrace_command(output_xvg),
        # --- TUI integration: selected parameters, units, provenance ---
        "index_file": index_file,
        "parameters": {
            "fit_group": fit_group, "calc_group": calc_group, "fit_group_used": fit_mode != "none",
            "reference": struct_ref, "reference_is_tpr": ref_is_tpr, "fit_mode": fit_mode,
            "mass_weighted": mass, "pbc": pbc, "time_window": window.as_dict(),
        },
        "units": {"time": "ns", "rmsd": "nm", "time_window_input": "ns", "time_passed_to_gmx": "ns (-tu ns)"},
        "gmx_command": list(cmd),
        "gmx_stdin_groups": stdin_groups,
        "gmx_stdin_input": stdin_tokens,
        "group_resolution": {"fit_group": fit_ref.as_dict(), "calc_group": calc_ref.as_dict()},
        "time_coverage": coverage,
        "warnings": warnings,
        "statistical_reality_check": reality,
    }

    return AnalysisResult(
        analysis_name="RMSD",
        output_xvg=output_xvg,
        output_csv=output_csv,
        data=df,
        metadata=metadata,
        statistics=summary
    )
