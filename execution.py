"""Configured GROMACS command shared by the supplied analyses."""
import os
import shlex
def gmx_command():
    command = shlex.split(os.environ.get("MD_GMX_COMMAND", "gmx"))
    if not command:
        raise ValueError("Choose a GROMACS command.")
    if any("wsl" in part.lower() for part in command[:1]):
        raise ValueError("Run the TUI inside WSL with Linux paths for analysis and simulation.")
    return command
