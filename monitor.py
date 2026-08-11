from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from metrics import PerformanceEstimator, calculate_eta_seconds
from models import MonitorSnapshot, RunStatus, SimulationConfig, SimulationState
from parser import parse_log_file, parse_mdp_file


class SimulationMonitor:
    """Coordinate log parsing and metrics across repeated observations."""

    def __init__(self, log_path: Path, mdp_path: Path | None = None) -> None:
        self.log_path = log_path
        self.mdp_path = mdp_path
        self._estimator = PerformanceEstimator()

    def _read_config(self) -> tuple[SimulationConfig | None, str | None]:
        if self.mdp_path is None:
            return None, None
        try:
            return parse_mdp_file(self.mdp_path), None
        except FileNotFoundError:
            return None, f"MDP file not found: {self.mdp_path}"
        except (OSError, ValueError) as error:
            return None, f"Could not read MDP file: {error}"

    def snapshot(self, observed_at: float | None = None) -> MonitorSnapshot:
        observed_at = time.monotonic() if observed_at is None else observed_at
        config, config_error = self._read_config()
        try:
            state = parse_log_file(self.log_path, config)
        except FileNotFoundError:
            state = SimulationState(
                status=RunStatus.WAITING,
                total_steps=config.total_steps if config else None,
                dt_ps=config.dt_ps if config else None,
                error_message=f"Waiting for log file: {self.log_path}",
            )
        except (OSError, ValueError) as error:
            state = SimulationState(
                status=RunStatus.ERROR,
                total_steps=config.total_steps if config else None,
                dt_ps=config.dt_ps if config else None,
                error_message=f"Could not read log file: {error}",
            )

        if config_error and state.error_message is None:
            state = replace(state, error_message=config_error)

        self._estimator.add_sample(observed_at, state.simulated_time_ps)
        live_speed = self._estimator.speed_ns_day
        if state.status is RunStatus.COMPLETE and state.final_performance_ns_day is not None:
            speed = state.final_performance_ns_day
            source = "GROMACS final"
        else:
            speed = live_speed
            source = "live estimate" if live_speed is not None else None

        eta = (
            0.0
            if state.status is RunStatus.COMPLETE
            else calculate_eta_seconds(
                state.current_step,
                state.total_steps,
                state.dt_ps,
                speed,
            )
        )
        return MonitorSnapshot(
            state=state,
            speed_ns_day=speed,
            eta_seconds=eta,
            speed_source=source,
        )
