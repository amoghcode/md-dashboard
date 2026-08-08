from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RunStatus(str, Enum):
    """High-level status inferred from an mdrun log."""

    WAITING = "waiting"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class SimulationConfig:
    total_steps: int | None = None
    dt_ps: float | None = None


@dataclass(frozen=True)
class SimulationState:
    status: RunStatus
    current_step: int | None = None
    simulated_time_ps: float | None = None
    total_steps: int | None = None
    dt_ps: float | None = None
    temperature_k: float | None = None
    pressure_bar: float | None = None
    warning_count: int = 0
    recent_warnings: tuple[str, ...] = field(default_factory=tuple)
    final_performance_ns_day: float | None = None
    error_message: str | None = None
    last_log_update: datetime | None = None

    @property
    def simulated_time_ns(self) -> float | None:
        if self.simulated_time_ps is None:
            return None
        return self.simulated_time_ps / 1000.0

    @property
    def progress_percent(self) -> float | None:
        if self.current_step is None or not self.total_steps or self.total_steps <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * self.current_step / self.total_steps))
