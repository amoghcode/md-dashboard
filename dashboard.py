from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from models import MonitorSnapshot, RunStatus
from monitor import SimulationMonitor
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


STATUS_STYLE = {
    RunStatus.WAITING: "yellow",
    RunStatus.STARTING: "cyan",
    RunStatus.RUNNING: "bright_blue",
    RunStatus.COMPLETE: "green",
    RunStatus.ERROR: "bold red",
}


def _number(value: float | int | None, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:,.{precision}f}{suffix}"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculating…"
    seconds = max(0, round(seconds))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def build_dashboard(snapshot: MonitorSnapshot, log_path: Path) -> Group:
    state = snapshot.state
    status = Text(state.status.value.upper(), style=STATUS_STYLE[state.status])

    summary = Table.grid(expand=True, padding=(0, 2))
    summary.add_column(style="bold", ratio=1)
    summary.add_column(ratio=2)
    summary.add_column(style="bold", ratio=1)
    summary.add_column(ratio=2)
    summary.add_row("Status", status, "Log", str(log_path))
    step = (
        f"{state.current_step:,} / {state.total_steps:,}"
        if state.current_step is not None and state.total_steps is not None
        else _number(state.current_step)
    )
    last_update = (
        state.last_log_update.strftime("%Y-%m-%d %H:%M:%S")
        if state.last_log_update
        else "—"
    )
    summary.add_row("Step", step, "Last update", last_update)

    progress = state.progress_percent or 0.0
    progress_grid = Table.grid(expand=True)
    progress_grid.add_column(ratio=8)
    progress_grid.add_column(width=10, justify="right")
    progress_grid.add_row(ProgressBar(total=100, completed=progress), f"{progress:.2f}%")

    metrics = Table(show_header=False, box=None, expand=True, padding=(0, 2))
    metrics.add_column(style="bold cyan")
    metrics.add_column()
    metrics.add_column(style="bold cyan")
    metrics.add_column()
    speed = (
        f"{snapshot.speed_ns_day:.2f} ns/day ({snapshot.speed_source})"
        if snapshot.speed_ns_day is not None
        else "calculating…"
    )
    metrics.add_row(
        "Simulated time",
        _number(state.simulated_time_ns, " ns", 3),
        "Speed",
        speed,
    )
    metrics.add_row(
        "Temperature",
        _number(state.temperature_k, " K"),
        "ETA",
        format_duration(snapshot.eta_seconds),
    )
    metrics.add_row(
        "Pressure",
        _number(state.pressure_bar, " bar"),
        "Warnings",
        str(state.warning_count),
    )

    panels: list[Any] = [
        Panel(
            summary,
            title="GROMACS Simulation Monitor",
            border_style=STATUS_STYLE[state.status],
        ),
        Panel(progress_grid, title="Progress"),
        Panel(metrics, title="Current metrics"),
    ]
    if state.error_message:
        style = "red" if state.status is RunStatus.ERROR else "yellow"
        panels.append(Panel(state.error_message, title="Notice", border_style=style))
    if state.recent_warnings:
        warnings = "\n".join(f"• {warning}" for warning in state.recent_warnings)
        panels.append(Panel(warnings, title="Recent warnings", border_style="yellow"))
    elif not state.error_message:
        panels.append(Panel("No warnings detected", title="Warnings", border_style="green"))
    panels.append(
        Text(
            f"Refreshed {datetime.now().astimezone():%H:%M:%S}  •  Ctrl+C to quit",
            style="dim",
        )
    )
    return Group(*panels)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor a GROMACS simulation log.")
    parser.add_argument("--log", "-l", required=True, type=Path)
    parser.add_argument(
        "--mdp",
        "-m",
        type=Path,
        help="Optional MDP file; nsteps and dt otherwise come from the log",
    )
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument("--json", action="store_true", help="Print one JSON snapshot and exit")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.refresh < 0.2:
        raise SystemExit("--refresh must be at least 0.2 seconds")

    monitor = SimulationMonitor(args.log, args.mdp)
    console = Console()
    if args.json:
        print(json.dumps(asdict(monitor.snapshot()), default=_json_default, indent=2))
        return
    if args.once:
        console.print(build_dashboard(monitor.snapshot(), args.log))
        return

    try:
        with Live(
            build_dashboard(monitor.snapshot(), args.log),
            console=console,
            refresh_per_second=4,
        ) as live:
            while True:
                time.sleep(args.refresh)
                live.update(build_dashboard(monitor.snapshot(), args.log), refresh=True)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
