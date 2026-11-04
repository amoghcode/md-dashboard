"""Read trajectory metadata before launching scientific calculations."""
from pathlib import Path
import math
def validate_analysis(request):
    import MDAnalysis as mda
    begin, end = float(request["begin"]), float(request["end"])
    if not math.isfinite(begin) or not math.isfinite(end) or begin < 0 or (end != -1 and end < begin):
        raise ValueError("Choose a finite, ordered time window in ns.")
    for key in ("topology", "trajectory"):
        if not Path(request[key]).is_file():
            raise ValueError("Missing "+key+" file")
    universe = mda.Universe(request["topology"], request["trajectory"])
    try:
        first = universe.trajectory[0].time / 1000
        last = universe.trajectory[-1].time / 1000
        if begin > last or (end != -1 and end < first):
            raise ValueError("Requested interval has no overlap with trajectory coverage.")
        warnings = ["Equal atom counts do not prove atom ordering. Verify that these files originate from the same run."]
        if begin < first or (end != -1 and end > last):
            warnings.append("Requested interval exceeds available trajectory coverage.")
        if request.get("index"):
            from analysis_backend.gmx_options import read_index_groups
            groups = read_index_groups(request["index"])
            if request["group"] not in groups:
                raise ValueError("Selected group is absent from the index.")
            atoms = groups[request["group"]]
            if not atoms or min(atoms) < 1 or max(atoms) > len(universe.atoms):
                raise ValueError("Index group contains invalid atom numbers.")
        else:
            mapping = {"Backbone":"backbone", "Protein":"protein", "C-alpha":"protein and name CA", "System":"all"}
            selection = mapping.get(request["group"])
            if selection is None:
                raise ValueError("Use an index file for this group.")
            if not len(universe.select_atoms(selection)):
                raise ValueError("Selected group is empty for this system.")
        return {"atoms": len(universe.atoms), "frames": len(universe.trajectory),
                "first_ns": first, "last_ns": last, "warnings": warnings}

    finally:
        universe.trajectory.close()
