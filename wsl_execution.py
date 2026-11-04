"""Run Python and GROMACS together in the configured WSL distribution."""
import json
import subprocess
from pathlib import Path
def split_command(tokens):
    if "--exec" in tokens:
        i=tokens.index("--exec")
        return tokens[:i],tokens[i+1:]
    if "-e" in tokens:
        i=tokens.index("-e")
        return tokens[:i],tokens[i+1:]
    raise ValueError("Use wsl --distribution NAME --exec gmx.")
def translate(prefix,path):
    return subprocess.check_output(prefix+["--exec","wslpath","-a","-u",str(path)],text=True,timeout=15).strip()
def worker_command(tokens, request_path, env):
    prefix,gmx=split_command(tokens)
    if not gmx: raise ValueError("Missing Linux GROMACS command")
    request=json.loads(request_path.read_text())
    for key in ("topology","trajectory","index","source","resume"):
        if request.get(key): request[key]=translate(prefix,request[key])
    request_path.write_text(json.dumps(request,indent=2))
    worker=translate(prefix,Path(__file__).with_name("workflow_worker.py").resolve())
    linux_request=translate(prefix,request_path)
    import shlex
    return prefix+["--exec","env","MD_GMX_COMMAND="+shlex.join(gmx),"python3",worker,linux_request]
