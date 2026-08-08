from dataclasses import dataclass

@dataclass
class SimulationState:
    current_step: int
    simulated_time_ps: float
    total_steps: int

    @property
    def simulated_time_ns(self) -> float:
        return self.simulated_time_ps / 1000

    @property
    def progress_percent(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return 100 * self.current_step / self.total_steps