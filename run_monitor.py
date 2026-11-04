"""Reuse the existing dashboard while following a launched simulation."""
import json
from pathlib import Path
from monitor import SimulationMonitor

class LaunchedRunMonitor:
    def __init__(self, job_folder, inputs_folder):
        self.job_folder = Path(job_folder)
        self.inputs_folder = Path(inputs_folder)
        self.stage = None
        self.monitor = None

    def snapshot(self):
        try:
            status = json.loads((self.job_folder / "stage.json").read_text())
        except (OSError, ValueError):
            return None
        stage = status.get("stage")
        if stage not in ("em", "npt_eq", "npt_prod"):
            return None
        if stage != self.stage:
            self.stage = stage
            self.monitor = SimulationMonitor(self.inputs_folder / (stage + ".log"),
                                             self.inputs_folder / (stage + ".mdp"))
        return stage, self.monitor.snapshot()
