import argparse
from pathlib import Path

from parser import parse_state
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Monitor a GROMACS simulation log."
    )
    argument_parser.add_argument("--log", required=True, type=Path)
    argument_parser.add_argument("--mdp", required=True, type=Path)
    args = argument_parser.parse_args()

    state = parse_state(args.log, args.mdp)

    table = Table(show_header=False)
    table.add_row("Step", f"{state.current_step:,} / {state.total_steps:,}")
    table.add_row("Simulated time", f"{state.simulated_time_ns:.3f} ns")
    table.add_row("Progress", f"{state.progress_percent:.2f}%")

    console.print(Panel(table, title="GROMACS Simulation"))


if __name__ == "__main__":
    main()