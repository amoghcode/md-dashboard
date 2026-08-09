from models import RunStatus, SimulationConfig
from parser import parse_log_text, parse_mdp_text


def test_parse_mdp_ignores_comments_and_spacing() -> None:
    config = parse_mdp_text("dt = 0.002 ; ps\n nsteps=50000000 ; production\n")
    assert config.dt_ps == 0.002
    assert config.total_steps == 50_000_000


def test_latest_complete_step_and_energy_are_parsed() -> None:
    log = """
Input Parameters:
  dt = 0.002
  nsteps = 50000000
           Step           Time
            100      0.20000
   Energies (kJ/mol)
    Kinetic En.   Total Energy  Conserved En.    Temperature Pres. DC (bar)
    1.30000e+04   -2.80000e+04    1.20000e+06    2.93100e+02    1.00000e+00
 Pressure (bar)   Constr. rmsd
    1.20000e+00    0.00000e+00
           Step           Time
            200      0.40000
   Energies (kJ/mol)
    Kinetic En.   Total Energy  Conserved En.    Temperature Pres. DC (bar)
    1.31775e+04   -2.84783e+04    1.21640e+06    2.94015e+02   -1.48420e+02
 Pressure (bar)   Constr. rmsd
    5.97514e+02    0.00000e+00
"""
    state = parse_log_text(log)
    assert state.status is RunStatus.RUNNING
    assert state.current_step == 200
    assert state.simulated_time_ps == 0.4
    assert state.simulated_time_ns == 0.0004
    assert state.temperature_k == 294.015
    assert state.pressure_bar == 597.514


def test_incomplete_final_step_record_is_ignored() -> None:
    state = parse_log_text(
        "Step Time\n100 0.2\nStep Time\n",
        SimulationConfig(total_steps=1_000, dt_ps=0.002),
    )
    assert state.current_step == 100
    assert state.progress_percent == 10.0


def test_completed_log_uses_final_performance() -> None:
    log = """
Step Time
50000000 100000.0
Performance: 232.662 0.103
Finished mdrun on rank 0
"""
    state = parse_log_text(log, SimulationConfig(50_000_000, 0.002))
    assert state.status is RunStatus.COMPLETE
    assert state.progress_percent == 100.0
    assert state.final_performance_ns_day == 232.662


def test_warning_and_fatal_error_detection() -> None:
    state = parse_log_text(
        "LINCS WARNING at step 25\nWARNING: constraint deviation\nFatal error: stopped\n"
    )
    assert state.status is RunStatus.ERROR
    assert state.warning_count == 2
    assert state.error_message == "Fatal error"


def test_empty_log_is_waiting() -> None:
    assert parse_log_text("").status is RunStatus.WAITING
