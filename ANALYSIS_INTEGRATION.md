# Analysis integration
Source: supplied gromacs_analysis_final/analysis_backend; adapted to configured execution.py command.
Run the TUI within Linux/WSL for GROMACS jobs. Native Windows is supported only with native GROMACS; cross-WSL command prefixes are explicitly rejected.
Each job creates a unique results directory, request.json and execution.log. Analysis adds numerical outputs, metadata and an HTML graph.
The simulation form takes the exercise input folder, copies it and runs three stages. Short mode changes nsteps in the copies. Warnings are not automatically overridden.
Current limits: no checkpoint resume UI, true percentage, automatic trajectory repair, remote execution, or independent scientific validation. Stop terminates the job process tree; simulation checkpoints are not guaranteed on forced stop. Do not interpret a smoke test as validated production science.
