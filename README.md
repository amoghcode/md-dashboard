# MD Dashboard Prototype

A Textual terminal prototype for molecular-dynamics project setup and GROMACS
run monitoring. The overview shows the planned scientific workflow and checks
the current Python and GROMACS environment. Separate tabs provide setup
guidance, GROMACS command verification, monitoring for existing `mdrun` logs,
and an analysis workspace placeholder for methods that will be supplied later.

## Features

- Tabbed Textual interface with overview, environment, setup, GROMACS, and monitor screens
- Environment overview for Python, project packages, GROMACS, and WSL2
- Guided setup instructions for missing Python dependencies or GROMACS
- Verify and save a GROMACS executable command in per-user settings
- Workflow preview from project setup through analysis
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

On Windows, run from Ubuntu on WSL2 when you need GROMACS. The environment tab
can inspect Python packages even when GROMACS is absent. The setup tab can
create or reuse the project `.venv` and install its requirements after an
explicit confirmation. GROMACS installation is guided from the system terminal;
shell startup files are never edited.

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

1. Confirm the Overview lists the seven workflow stages.
2. Open Environment and refresh the checks.
3. Open Setup and review the platform-specific guidance. The Python setup action
   asks for confirmation before installing into `.venv`.
4. If GROMACS is installed, enter `gmx` (or its executable path), verify it, and
   optionally save the verified command.
5. Open Analysis and select Structural to reveal RMSD, RMSF, and RoG buttons.
   Confirm that selecting these buttons or Thermodynamic has no effect.
6. Open Monitor, enter a GROMACS log path and optional MDP path, then start
   monitoring. Use a real existing log to see parsed progress and metrics.
7. Press `Ctrl+Q` to exit.

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

- The prototype inspects and documents setup but does not run package managers
  or edit shell profiles automatically.
- It monitors existing GROMACS logs; it does not prepare or launch simulations.
- Temperature and pressure are parsed from log text.
- A static partial log cannot reveal speed. Live speed requires observing it
  grow.
- An unfinished log with progress is shown as running even if its external
  process stopped between refreshes.
