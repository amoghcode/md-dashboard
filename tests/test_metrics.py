import pytest

from metrics import PerformanceEstimator, calculate_eta_seconds, calculate_speed_ns_day
from monitor import SimulationMonitor


def test_speed_converts_ps_and_seconds_to_ns_per_day() -> None:
    assert calculate_speed_ns_day(0, 1_000, 1) == 86_400


def test_invalid_speed_samples_return_none() -> None:
    assert calculate_speed_ns_day(1_000, 1_000, 1) is None
    assert calculate_speed_ns_day(0, 1_000, 0) is None


def test_eta_uses_remaining_steps_dt_and_speed() -> None:
    eta = calculate_eta_seconds(25_000_000, 50_000_000, 0.002, 100)
    assert eta == pytest.approx(43_200)


def test_estimator_needs_two_advancing_samples_and_handles_restart() -> None:
    estimator = PerformanceEstimator()
    estimator.add_sample(10, 1_000)
    estimator.add_sample(11, 1_000)
    assert estimator.speed_ns_day is None
    estimator.add_sample(12, 2_000)
    assert estimator.speed_ns_day == pytest.approx(43_200)
    estimator.add_sample(13, 100)
    assert estimator.speed_ns_day is None


def test_live_monitor_calculates_speed_and_eta(tmp_path) -> None:
    log = tmp_path / "npt_prod.log"
    mdp = tmp_path / "npt_prod.mdp"
    mdp.write_text("dt = 1\nnsteps = 2000\n", encoding="utf-8")
    log.write_text("Step Time\n1000 1000.0\n", encoding="utf-8")
    monitor = SimulationMonitor(log, mdp)
    assert monitor.snapshot(observed_at=10).speed_ns_day is None
    log.write_text("Step Time\n1500 1500.0\n", encoding="utf-8")
    snapshot = monitor.snapshot(observed_at=11)
    assert snapshot.speed_ns_day == 43_200
    assert snapshot.eta_seconds == 1
    assert snapshot.speed_source == "live estimate"


def test_missing_log_is_reported_without_crashing(tmp_path) -> None:
    mdp = tmp_path / "npt_prod.mdp"
    mdp.write_text("dt = 0.002\nnsteps = 1000\n", encoding="utf-8")
    snapshot = SimulationMonitor(tmp_path / "missing.log", mdp).snapshot(observed_at=1)
    assert snapshot.state.status.value == "waiting"
    assert snapshot.state.total_steps == 1_000
    assert "Waiting for log file" in (snapshot.state.error_message or "")
