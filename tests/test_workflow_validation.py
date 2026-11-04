"""Regression: Universe is not a context manager; always close its reader."""
import sys
from types import SimpleNamespace
import pytest
from workflow_validation import validate_analysis

@pytest.mark.parametrize("empty_group", [False, True])
def test_reader_closed_on_success_and_validation_error(tmp_path, monkeypatch, empty_group):
    class Reader:
        closed = False
        def __getitem__(self, index):
            return SimpleNamespace(time=0 if index == 0 else 100)
        def __len__(self):
            return 11
        def close(self):
            self.closed = True
    reader = Reader()
    universe = SimpleNamespace(
        trajectory=reader, atoms=[1,2,3],
        select_atoms=lambda selection: [] if empty_group else [1,2,3])
    monkeypatch.setitem(sys.modules, "MDAnalysis", SimpleNamespace(Universe=lambda *args: universe))
    topology=tmp_path/"run.tpr"
    trajectory=tmp_path/"run.xtc"
    topology.touch()
    trajectory.touch()
    request=dict(topology=str(topology),trajectory=str(trajectory),group="System",begin="0",end="-1")
    if empty_group:
        with pytest.raises(ValueError,match="empty"):
            validate_analysis(request)
    else:
        result=validate_analysis(request)
        assert result["last_ns"] == .1
        assert result["frames"] == 11
    assert reader.closed
