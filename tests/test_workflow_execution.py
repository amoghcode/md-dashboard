import pytest
from wsl_execution import split_command, translate
def test_explicit_distribution_preserved():
    assert split_command(["wsl","-d","Ubuntu","--exec","gmx"]) == (["wsl","-d","Ubuntu"],["gmx"])
def test_ambiguous_command_rejected():
    with pytest.raises(ValueError): split_command(["wsl","gmx"])
def test_path_is_separate_argument(monkeypatch):
    def capture(args,**kwargs):
        assert args[-1] == r"C:\lab data\input.tpr"
        assert args[:3] == ["wsl","-d","Lab"]
        return "/mnt/c/lab data/input.tpr\n"
    monkeypatch.setattr("subprocess.check_output",capture)
    assert translate(["wsl","-d","Lab"],r"C:\lab data\input.tpr") == "/mnt/c/lab data/input.tpr"
