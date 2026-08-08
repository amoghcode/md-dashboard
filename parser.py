from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from models import RunStatus, SimulationConfig, SimulationState


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
STEP_TIME_PATTERN = re.compile(
    rf"^\s*Step\s+Time\s*$\s*^\s*(\d+)\s+({FLOAT})\s*$",
    re.MULTILINE,
)
PERFORMANCE_PATTERN = re.compile(rf"^Performance:\s+({FLOAT})\b", re.MULTILINE)
FINISHED_PATTERN = re.compile(r"Finished mdrun", re.IGNORECASE)
FATAL_PATTERN = re.compile(
    r"Fatal error|Segmentation fault|core dumped|MPI_ABORT",
    re.IGNORECASE,
)
WARNING_PATTERN = re.compile(r"\b(?:LINCS\s+)?WARNING\b", re.IGNORECASE)


def parse_mdp_text(text: str) -> SimulationConfig:
    """Extract nsteps and dt while respecting MDP semicolon comments."""

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        if key in {"nsteps", "dt"} and value.strip():
            values[key] = value.strip().split()[0]

    return SimulationConfig(
        total_steps=int(values["nsteps"]) if "nsteps" in values else None,
        dt_ps=float(values["dt"]) if "dt" in values else None,
    )


def parse_mdp_file(path: Path) -> SimulationConfig:
    return parse_mdp_text(path.read_text(encoding="utf-8", errors="replace"))


def read_total_steps(mdp_path: Path) -> int:
    """Backward-compatible helper retained from the first implementation."""

    total_steps = parse_mdp_file(mdp_path).total_steps
    if total_steps is None:
        raise ValueError(f"Could not find nsteps in {mdp_path}")
    return total_steps


def _config_from_log(text: str) -> SimulationConfig:
    """Read settings from GROMACS's Input Parameters when no MDP is available."""

    nsteps = re.search(r"^\s*nsteps\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    dt = re.search(rf"^\s*dt\s*=\s*({FLOAT})\s*$", text, re.MULTILINE)
    return SimulationConfig(
        total_steps=int(nsteps.group(1)) if nsteps else None,
        dt_ps=float(dt.group(1)) if dt else None,
    )


def _parse_energy_block(segment: str) -> dict[str, float]:
    """Parse the first energy table following the latest Step/Time record."""

    lines = segment.splitlines()
    try:
        index = next(i for i, line in enumerate(lines) if "Energies (kJ/mol)" in line) + 1
    except StopIteration:
        return {}

    energies: dict[str, float] = {}
    while index + 1 < len(lines):
        header_line, value_line = lines[index], lines[index + 1]
        if not header_line.strip() or not value_line.strip():
            break
        value_matches = list(re.finditer(FLOAT, value_line))
        if not value_matches:
            break
        # GROMACS renders energy tables as right-aligned, 15-character fields.
        # Long adjacent labels can therefore have only one separating space, so
        # splitting headers on whitespace is not reliable. Each numeric field's
        # end column gives us the corresponding header field boundary.
        fields = [
            (
                header_line[max(0, value_match.end() - 15) : value_match.end()].strip(),
                float(value_match.group(0)),
            )
            for value_match in value_matches
        ]
        if any(not header for header, _ in fields):
            break
        energies.update(fields)
        index += 2
    return energies


def parse_log_text(
    text: str,
    config: SimulationConfig | None = None,
) -> SimulationState:
    """Parse a complete, partial, empty, or still-growing GROMACS log."""

    fallback = _config_from_log(text)
    effective = SimulationConfig(
        total_steps=(
            config.total_steps
            if config is not None and config.total_steps is not None
            else fallback.total_steps
        ),
        dt_ps=(
            config.dt_ps if config is not None and config.dt_ps is not None else fallback.dt_ps
        ),
    )

    warnings = tuple(
        line.strip() for line in text.splitlines() if WARNING_PATTERN.search(line)
    )
    fatal = FATAL_PATTERN.search(text)
    finished = bool(FINISHED_PATTERN.search(text))
    step_matches = list(STEP_TIME_PATTERN.finditer(text))

    if fatal:
        status = RunStatus.ERROR
    elif finished:
        status = RunStatus.COMPLETE
    elif step_matches:
        status = RunStatus.RUNNING
    elif text.strip():
        status = RunStatus.STARTING
    else:
        status = RunStatus.WAITING

    current_step: int | None = None
    simulated_time_ps: float | None = None
    temperature_k: float | None = None
    pressure_bar: float | None = None
    if step_matches:
        latest = step_matches[-1]
        current_step = int(latest.group(1))
        simulated_time_ps = float(latest.group(2))
        energies = _parse_energy_block(text[latest.end() :])
        temperature_k = energies.get("Temperature")
        pressure_bar = energies.get("Pressure (bar)")

    performances = list(PERFORMANCE_PATTERN.finditer(text))
    final_performance = float(performances[-1].group(1)) if performances else None

    return SimulationState(
        status=status,
        current_step=current_step,
        simulated_time_ps=simulated_time_ps,
        total_steps=effective.total_steps,
        dt_ps=effective.dt_ps,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        warning_count=len(warnings),
        recent_warnings=warnings[-5:],
        final_performance_ns_day=final_performance,
        error_message=fatal.group(0) if fatal else None,
    )


def parse_log_file(path: Path, config: SimulationConfig | None = None) -> SimulationState:
    state = parse_log_text(path.read_text(encoding="utf-8", errors="replace"), config)
    modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    return replace(state, last_log_update=modified)


def parse_state(log_path: Path, mdp_path: Path) -> SimulationState:
    """Backward-compatible file entry point used by early versions."""

    return parse_log_file(log_path, parse_mdp_file(mdp_path))
