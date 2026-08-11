# GROMACS Live Monitor

A terminal dashboard that turns a complete or growing GROMACS `mdrun` log into a
readable simulation summary. It was built for the supplied 30 wt% H2O2 exercise,
but no exercise paths or values are hard-coded.

## Features

- Current step, simulated time, and progress percentage
- Live speed derived from simulated-time and observed wall-time changes
- Estimated completion time (ETA)
- Official GROMACS performance for completed logs
- Latest temperature and pressure from the log's energy block
- Warning and LINCS warning detection
- Waiting, starting, running, complete, and error states
- Safe behavior for missing, empty, and partially written logs
- Last log modification time
- Live dashboard, one-shot output, and JSON output

## Installation in Ubuntu/WSL2

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reference log can be monitored without installing GROMACS. GROMACS is only
needed to produce a new live log.

## Usage

Completed reference run:

```bash
python dashboard.py \
  --log ~/student_package/exercise_system/reference_output/npt_prod.log \
  --mdp ~/student_package/exercise_system/input/npt_prod.mdp \
  --once
```

Live run (omit `--once`):

```bash
python dashboard.py \
  --log ~/student_package/exercise_system/input/npt_prod.log \
  --mdp ~/student_package/exercise_system/input/npt_prod.mdp
```

Press `Ctrl+C` to exit. Use `--refresh 5` to change the two-second polling
interval. The minimum is 0.2 seconds.

The MDP is optional because a normal GROMACS log includes `nsteps` and `dt` in
its Input Parameters section:

```bash
python dashboard.py --log /path/to/npt_prod.log --once
```

Machine-readable output:

```bash
python dashboard.py --log /path/to/npt_prod.log --mdp /path/to/npt_prod.mdp --json
```

## Metric definitions

GROMACS log time is in picoseconds, converted with `1 ns = 1000 ps`.

```text
progress = current step / nsteps × 100
live speed = simulated-time change (ns) / wall-time change (days)
```

Speed and ETA show `calculating…` until a live log has advanced between at least
two observations. Completed logs use the final `Performance` value written by
GROMACS. Temperature and pressure come from the energy block following the latest
complete Step/Time record, so they may briefly be unavailable while that block is
being written.

## Tests

```bash
python -m pytest
```

Tests cover completed, partial, empty, missing, warning, and fatal-error logs,
plus unit conversion, speed, ETA, and restarted simulations.

## Files

```text
dashboard.py   Rich terminal UI and command-line interface
monitor.py     Repeated observations and live metric coordination
parser.py      MDP, progress, energy, warning, and status parsing
metrics.py     Speed and ETA calculations
models.py      Typed application state
tests/         Parser and calculation tests
```

## Limitations

- This tool monitors an existing log; it does not start or terminate GROMACS.
- Temperature and pressure are parsed from log text, not the binary `.edr` file.
- A static partial log cannot reveal speed. Live speed requires observing it grow.
- A log alone cannot prove process liveness. An unfinished log with progress is
  shown as running even if its external process was stopped between refreshes.
