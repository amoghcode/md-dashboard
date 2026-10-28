# MD Dashboard Prototype

A Textual terminal prototype for molecular-dynamics project setup and GROMACS
run monitoring. The overview summarizes the current Python and GROMACS
environment. Separate tabs provide setup guidance, GROMACS command
verification, monitoring for existing `mdrun` logs, and an analysis workspace
placeholder for methods that will be supplied later.

## Features

- Tabbed Textual interface with a combined overview, setup, GROMACS, and monitor screens
- Environment overview for Python, project packages, GROMACS, and WSL2
- Install Python requirements in the project `.venv` after confirmation
- Install GROMACS after confirmation, choosing a CUDA source build for compatible NVIDIA GPUs when supported
- Verify and save a GROMACS executable command in per-user settings
- Analysis tab with Structural and Thermodynamic categories; Structural reveals inert RMSD, RMSF, and RoG placeholder buttons
- Current step, simulated time, progress percentage, speed, and ETA
- Latest temperature and pressure parsed from the GROMACS log
- Warning and LINCS warning detection
- Waiting, starting, running, complete, and error states
- Live dashboard, one-shot output, and JSON output
- Safe handling of missing, empty, and partially written logs

## Install and run

### Windows PowerShell

From the repository folder, create an isolated Python environment, install the
dependencies, and launch the TUI:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dashboard.py
```

The first launch opens the Overview tab. Use the tab bar to move between
screens. Press `Ctrl+Q` to quit.

### Ubuntu or WSL2

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, the app can install GROMACS into an existing default Ubuntu or
Debian WSL distribution. On Debian or Ubuntu, the Setup tab installs the system
`gromacs` package after confirmation. When an NVIDIA GPU with CUDA compute
capability 5.0 or newer is visible inside an Ubuntu 24.04 or 26.04 x86-64
environment, Setup instead installs a compatible NVIDIA CUDA toolkit and builds
GROMACS 2026.4 from its official source archive. CUDA 12.9 is selected for older
NVIDIA architectures or drivers; newer hardware uses a compatible CUDA 13
toolkit. On Windows, the GPU and driver must also be available inside WSL2. The
CUDA build can take several minutes and downloads several gigabytes. Other
systems use the repository package or show platform guidance. Intel and AMD GPU
builds are not installed automatically yet. Python requirements are installed
separately in the project `.venv` after confirmation. The installer does not
edit shell startup files or change GPU drivers.

## Usage

Launch the prototype:

```bash
python dashboard.py
```

Run a one-shot environment check or get machine-readable output:

```bash
python dashboard.py --check-environment
python dashboard.py --check-environment --json
```

The standard-library-only inspector can run before Rich is installed:

```bash
python environment.py
python environment.py --json
```

Monitor a completed reference run:

```bash
python dashboard.py \
  --log ~/student_package/exercise_system/reference_output/npt_prod.log \
  --mdp ~/student_package/exercise_system/input/npt_prod.mdp \
  --once
```

Monitor a live run by omitting `--once`:

```bash
python dashboard.py \
  --log ~/student_package/exercise_system/input/npt_prod.log \
  --mdp ~/student_package/exercise_system/input/npt_prod.mdp
```

Press `Ctrl+C` to exit. Use `--refresh 5` to change the two-second polling
interval. The minimum is 0.2 seconds.

## Metric definitions

GROMACS log time is in picoseconds, converted with `1 ns = 1000 ps`.

```text
progress = current step / nsteps × 100
live speed = simulated-time change (ns) / wall-time change (days)
```

Speed and ETA show `calculating…` until a live log advances between at least two
observations. Completed logs use the final `Performance` value written by
GROMACS. Temperature and pressure come from the energy block following the latest
complete Step/Time record, so they may briefly be unavailable while that block
is being written.

## Test the prototype

Run the automated parser and metrics suite from the repository root:

```bash
python -m pytest -q tests --basetemp .pytest_cache/md-tui-run
```

On Windows, use the project interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests --basetemp .pytest_cache/md-tui-run
```

For a manual TUI smoke check, launch `python dashboard.py` and:

1. Confirm the Overview shows the status summary and scrollable environment checks.
2. Open Setup and review the platform-specific guidance. Python and GROMACS
   install actions require confirmation.
3. If GROMACS is installed, enter `gmx` (or its executable path), verify it, and
   optionally save the verified command.
4. Open Analysis and select Structural to reveal RMSD, RMSF, and RoG buttons.
   Confirm that selecting these buttons or Thermodynamic has no effect.
5. Open Monitor, enter a GROMACS log path and optional MDP path, then start
   monitoring. Use a real existing log to see parsed progress and metrics.
6. Press `Ctrl+Q` to exit.

The automated tests cover completed, partial, empty, missing, warning, and
fatal-error logs, plus unit conversion, speed, ETA, and restarted simulations.

## Files

```text
dashboard.py   Rich log dashboard and command-line interface
environment.py Python dependency and GROMACS inspection/configuration
tui.py         Textual app, environment screens, and setup guide
monitor.py     Repeated observations and live metric coordination
parser.py      MDP, progress, energy, warning, and status parsing
metrics.py     Speed and ETA calculations
models.py      Typed application state
tests/         Parser and calculation tests
```

## Limitations

- CPU GROMACS installation uses the configured Debian/Ubuntu repository package;
  CUDA builds use the pinned GROMACS 2026.4 source release. The installer does
  not edit shell profiles automatically.
- It monitors existing GROMACS logs; it does not prepare or launch simulations.
- Temperature and pressure are parsed from log text.
- A static partial log cannot reveal speed. Live speed requires observing it
  grow.
- An unfinished log with progress is shown as running even if its external
  process stopped between refreshes.
