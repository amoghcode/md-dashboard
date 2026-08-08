import re
from pathlib import Path

from models import SimulationState


STEP_TIME_PATTERN = re.compile(
    r"^\s*Step\s+Time\s*$"
    r"\s*"
    r"^\s*(\d+)\s+([0-9.eE+-]+)\s*$",
    re.MULTILINE,
)

NSTEPS_PATTERN = re.compile(
    r"^\s*nsteps\s*=\s*(\d+)",
    re.MULTILINE,
)


def read_total_steps(mdp_path: Path) -> int:
    text = mdp_path.read_text(encoding="utf-8", errors="replace")
    match = NSTEPS_PATTERN.search(text)

    if match is None:
        raise ValueError(f"Could not find nsteps in {mdp_path}")

    return int(match.group(1))


def parse_state(log_path: Path, mdp_path: Path) -> SimulationState:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(STEP_TIME_PATTERN.finditer(log_text))

    if not matches:
        raise ValueError(f"No complete Step/Time record found in {log_path}")

    latest = matches[-1]

    return SimulationState(
        current_step=int(latest.group(1)),
        simulated_time_ps=float(latest.group(2)),
        total_steps=read_total_steps(mdp_path),
    )