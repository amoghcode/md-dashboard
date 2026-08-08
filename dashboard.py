import argparse
from pathlib import Path

from parser import parse_state


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Monitor a GROMACS simulation log."
    )
    argument_parser.add_argument("--log", required=True, type=Path)
    argument_parser.add_argument("--mdp", required=True, type=Path)
    args = argument_parser.parse_args()

    state = parse_state(args.log, args.mdp)

    print(f"Step: {state.current_step:,} / {state.total_steps:,}")
    print(f"Time: {state.simulated_time_ns:.3f} ns")
    print(f"Progress: {state.progress_percent:.2f}%")


if __name__ == "__main__":
    main()