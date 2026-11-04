import environment
from installer_policy import build_script
def test_cpu_lab_policy():
    plan,error=environment._linux_gromacs_install_plan(apt_get="apt-get",distro_id="ubuntu",distro_name="Ubuntu",sudo="sudo",is_root=False)
    assert not error
    assert "2026.2-cpu" in plan.target
    assert plan.commands[0][0] == "bash"
    assert "apt-get install -y gromacs" not in plan.commands[0][-1]
def test_resource_and_compiler_gates():
    script=build_script("gpu","https://example.test","cuda-toolkit-12-9","12.9")
    for text in ("MemAvailable", "6291456", "sort -V", "nvcc -ccbin", "CMAKE_CXX_COMPILER", "build-info.txt"):
        assert text in script
    assert script.index("MemAvailable") < script.index("wget")
def test_wsl_uses_normal_user():
    plan,_=environment._wsl_gromacs_install_plan("wsl","ID=ubuntu")
    assert plan.commands[0][:2] == ("wsl","--exec")
    assert "--user" not in plan.commands[0]
