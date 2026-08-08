from __future__ import annotations

from collections import deque


SECONDS_PER_DAY = 86_400.0
PS_PER_NS = 1_000.0


def calculate_speed_ns_day(
    earlier_time_ps: float,
    later_time_ps: float,
    elapsed_seconds: float,
) -> float | None:
    """Calculate ns/day from simulated-time and observed wall-time changes."""

    simulated_ps = later_time_ps - earlier_time_ps
    if simulated_ps <= 0 or elapsed_seconds <= 0:
        return None
    return (simulated_ps / PS_PER_NS) * SECONDS_PER_DAY / elapsed_seconds


def calculate_eta_seconds(
    current_step: int | None,
    total_steps: int | None,
    dt_ps: float | None,
    speed_ns_day: float | None,
) -> float | None:
    if (
        current_step is None
        or total_steps is None
        or dt_ps is None
        or speed_ns_day is None
        or speed_ns_day <= 0
    ):
        return None
    remaining_ns = max(0, total_steps - current_step) * dt_ps / PS_PER_NS
    return remaining_ns / speed_ns_day * SECONDS_PER_DAY


class PerformanceEstimator:
    """Maintain a rolling live performance estimate from advancing samples."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._samples: deque[tuple[float, float]] = deque(maxlen=window_size)

    def add_sample(self, observed_at: float, simulated_time_ps: float | None) -> None:
        if simulated_time_ps is None:
            return
        if self._samples and simulated_time_ps < self._samples[-1][1]:
            self._samples.clear()  # the monitored simulation restarted
        if self._samples and simulated_time_ps == self._samples[-1][1]:
            return
        self._samples.append((observed_at, simulated_time_ps))

    @property
    def speed_ns_day(self) -> float | None:
        if len(self._samples) < 2:
            return None
        first_wall, first_sim = self._samples[0]
        last_wall, last_sim = self._samples[-1]
        return calculate_speed_ns_day(first_sim, last_sim, last_wall - first_wall)
