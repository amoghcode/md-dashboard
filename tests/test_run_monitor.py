import json
from run_monitor import LaunchedRunMonitor
def test_wait_stage_transition_and_completed_log(tmp_path):
    inputs=tmp_path/"inputs"
    inputs.mkdir()
    follower=LaunchedRunMonitor(tmp_path,inputs)
    assert follower.snapshot() is None
    for stage in ("em","npt_eq","npt_prod"):
        (tmp_path/"stage.json").write_text(json.dumps({"stage":stage}))
        name,snapshot=follower.snapshot()
        assert name==stage
        assert snapshot.state.status.value=="waiting"
        (inputs/(stage+".mdp")).write_text("nsteps = 5000\ndt = 0.002")
        (inputs/(stage+".log")).write_text("Step Time\n5000 10.0\nFinished mdrun")
        _,snapshot=follower.snapshot()
        assert snapshot.state.status.value=="complete"
def test_invalid_or_partial_status_is_ignored(tmp_path):
    follower=LaunchedRunMonitor(tmp_path,tmp_path)
    for text in ("{", '{"stage":"../../other"}'):
        (tmp_path/"stage.json").write_text(text)
        assert follower.snapshot() is None
