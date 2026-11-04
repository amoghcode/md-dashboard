"""Isolated worker; inherits the configured GROMACS command."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from execution import gmx_command
def main():
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text())
    def stage_status(stage):
        temp = request_path.parent / "stage.tmp"
        temp.write_text(json.dumps({"stage":stage}))
        temp.replace(request_path.parent / "stage.json")
    os.chdir(request_path.parent)
    if request.get("resume"):
        target = Path(request["resume"]).resolve()
        manifest = json.loads((target.parent/"request.json").read_text())
        if manifest.get("mode") != "Simulation":
            raise ValueError("Resume requires a simulation created by this application.")
        if not (target/"npt_prod.cpt").is_file() or not (target/"npt_prod.tpr").is_file():
            raise ValueError("Production checkpoint and original TPR are required.")
        stage_status("npt_prod")
        os.chdir(target)
        subprocess.run(gmx_command()+["mdrun","-deffnm","npt_prod","-s","npt_prod.tpr","-cpi","npt_prod.cpt","-append"],check=True)
        return
    if request["mode"] != "Simulation":
        from workflow_validation import validate_analysis
        validation = validate_analysis(request)
        Path("input-validation.json").write_text(json.dumps(validation, indent=2))
        print(validation, flush=True)
        from analysis_backend.rmsd import run_rmsd
        from analysis_backend.rmsf import run_rmsf
        from analysis_backend.radius_of_gyration import run_gyrate
        common = dict(tpr_file=request["topology"], traj_file=request["trajectory"], index_file=request["index"] or None, b=float(request["begin"]), e=float(request["end"]), output_prefix="result")
        if request["mode"] == "RMSD":
            result = run_rmsd(**common, fit_group=request["group"], calc_group=request["group"])
        elif request["mode"] == "RMSF":
            result = run_rmsf(**common, selection=request["group"])
        else:
            result = run_gyrate(**common, group=request["group"])
        Path("metadata.json").write_text(json.dumps(result.metadata, indent=2, default=str))
        from analysis_backend.plotting import generate_plotly_html
        generate_plotly_html([result], output_html="plot.html")
        print(result.statistics, flush=True)
        return
    source = Path(request["source"]).expanduser().resolve()
    for name in ("box_mixed.gro","system.top","em.mdp","npt_eq.mdp","npt_prod.mdp"):
        if not (source/name).is_file():
            raise ValueError("Missing prepared input: "+name)
    shutil.copytree(source, "inputs")
    os.chdir("inputs")
    if request["steps"]:
        steps = int(request["steps"])
        if steps <= 0: raise ValueError("Steps must be positive")
        for name in ("em","npt_eq","npt_prod"):
            p=Path(name+".mdp")
            p.write_text(re.sub(r"(?m)^\s*nsteps\s*=.*$", "nsteps = "+str(steps),p.read_text()))
    mode=request["device"].lower()
    if mode not in ("auto","cpu","gpu"): raise ValueError("Device must be auto, cpu or gpu")
    for stage, coordinates in (("em","box_mixed.gro"),("npt_eq","em.gro"),("npt_prod","npt_eq.gro")):
        stage_status(stage)
        print("Preparing "+stage,flush=True)
        args=["grompp","-f",stage+".mdp","-c",coordinates,"-p","system.top","-o",stage+".tpr"]
        if stage=="npt_prod": args += ["-t","npt_eq.cpt"]
        subprocess.run(gmx_command()+args,check=True)
        flags=[] if mode=="auto" or stage=="em" else ["-nb",mode]
        subprocess.run(gmx_command()+["mdrun","-deffnm",stage]+flags,check=True)
if __name__ == "__main__":
    main()
