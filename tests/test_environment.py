from types import SimpleNamespace

import environment
from environment import GromacsInstallPlan, GromacsInstallation


def test_linux_install_plan_uses_distro_package_manager_and_sudo() -> None:
    plan, problem = environment._linux_gromacs_install_plan(
        apt_get="/usr/bin/apt-get",
        distro_id="ubuntu",
        distro_name="Ubuntu 24.04 LTS",
        sudo="/usr/bin/sudo",
        is_root=False,
    )

    assert problem == ""
    assert plan == GromacsInstallPlan(
        commands=(
            ("/usr/bin/sudo", "/usr/bin/apt-get", "update"),
            ("/usr/bin/sudo", "/usr/bin/apt-get", "install", "-y", "gromacs"),
        ),
        probe_command=("gmx",),
        source="native",
        target="Ubuntu 24.04 LTS package repositories (system-wide)",
    )


def test_wsl_install_plan_targets_default_ubuntu_as_root() -> None:
    plan, problem = environment._wsl_gromacs_install_plan(
        "C:/Windows/System32/wsl.exe",
        'NAME="Ubuntu"\nID=ubuntu\n',
    )

    assert problem == ""
    assert plan is not None
    assert plan.commands == (
        ("C:/Windows/System32/wsl.exe", "--user", "root", "--exec", "apt-get", "update"),
        (
            "C:/Windows/System32/wsl.exe",
            "--user",
            "root",
            "--exec",
            "apt-get",
            "install",
            "-y",
            "gromacs",
        ),
    )
    assert plan.probe_command == ("C:/Windows/System32/wsl.exe", "--exec", "gmx")


def test_nvidia_query_accepts_supported_cuda_capability() -> None:
    assert environment._parse_nvidia_gpu_query("NVIDIA GeForce RTX 4070, 8.9, 580.65.06\n") == (
        environment.NvidiaGPU("NVIDIA GeForce RTX 4070", 8.9, (580, 65, 6))
    )
    assert environment._parse_nvidia_gpu_query("NVIDIA GeForce GTX 750, 5.2, 575.57.08\n") == (
        environment.NvidiaGPU("NVIDIA GeForce GTX 750", 5.2, (575, 57, 8))
    )


def test_nvidia_query_skips_gpus_below_gromacs_minimum() -> None:
    assert environment._parse_nvidia_gpu_query("NVIDIA GeForce GTX 480, 2.0, 580.65.06\n") is None


def test_nvidia_query_uses_oldest_compatible_gpu_for_mixed_systems() -> None:
    assert environment._parse_nvidia_gpu_query(
        "NVIDIA GeForce RTX 4070, 8.9, 580.65.06\nNVIDIA GeForce GTX 980, 5.2, 575.57.08\n"
    ) == environment.NvidiaGPU("NVIDIA GeForce GTX 980", 5.2, (575, 57, 8))


def test_cuda_plan_builds_gromacs_from_source_for_supported_ubuntu() -> None:
    plan, problem = environment._nvidia_cuda_gromacs_install_plan(
        source="WSL",
        runner=("wsl.exe", "--user", "root", "--exec"),
        distro_id="ubuntu",
        distro_version="24.04",
        architecture="x86_64",
        gpu=environment.NvidiaGPU("NVIDIA GeForce RTX 4070", 8.9, (580, 65, 6)),
    )

    assert problem == ""
    assert plan is not None
    assert plan.gpu_backend == "CUDA"
    assert plan.probe_command == (
        "wsl.exe",
        "--user",
        "root",
        "--exec",
        "/usr/local/bin/gmx",
    )
    script = plan.commands[0][-1]
    assert "apt-get install -y cuda-toolkit-13-0" in script
    assert "-DGMX_GPU=CUDA" in script
    assert f"gromacs-{environment.GROMACS_GPU_BUILD_VERSION}.tar.gz" in script
    assert "cuda-drivers" not in script


def test_cuda_plan_explains_unsupported_distro() -> None:
    plan, problem = environment._nvidia_cuda_gromacs_install_plan(
        source="native",
        runner=("sudo",),
        distro_id="ubuntu",
        distro_version="22.04",
        architecture="x86_64",
        gpu=environment.NvidiaGPU("NVIDIA GeForce RTX 4070", 8.9, (580, 65, 6)),
    )

    assert plan is None
    assert "Ubuntu 24.04 and 26.04" in problem


def test_cuda_plan_selects_toolkit_that_still_supports_older_compatible_gpu() -> None:
    plan, problem = environment._nvidia_cuda_gromacs_install_plan(
        source="native",
        runner=("sudo",),
        distro_id="ubuntu",
        distro_version="24.04",
        architecture="x86_64",
        gpu=environment.NvidiaGPU("NVIDIA GeForce GTX 980", 5.2, (575, 57, 8)),
    )

    assert problem == ""
    assert plan is not None
    assert "cuda-toolkit-12-9" in plan.commands[0][-1]
    assert "/usr/local/cuda-12.9/bin/nvcc" in plan.commands[0][-1]


def test_cuda_plan_uses_cuda_12_for_new_gpu_with_older_compatible_driver() -> None:
    plan, problem = environment._nvidia_cuda_gromacs_install_plan(
        source="native",
        runner=("sudo",),
        distro_id="ubuntu",
        distro_version="24.04",
        architecture="x86_64",
        gpu=environment.NvidiaGPU("NVIDIA GeForce RTX 4070", 8.9, (570, 124, 6)),
    )

    assert problem == ""
    assert plan is not None
    assert "cuda-toolkit-12-9" in plan.commands[0][-1]


def test_cuda_plan_requires_driver_compatible_with_selected_toolkit() -> None:
    plan, problem = environment._nvidia_cuda_gromacs_install_plan(
        source="WSL",
        runner=("wsl.exe", "--user", "root", "--exec"),
        distro_id="ubuntu",
        distro_version="24.04",
        architecture="x86_64",
        gpu=environment.NvidiaGPU("NVIDIA GeForce RTX 4070", 8.9, (520, 0)),
    )

    assert plan is None
    assert "Update the driver on the Windows host" in problem


def test_windows_plan_detects_gpu_inside_wsl_and_selects_cuda_build(monkeypatch) -> None:
    monkeypatch.setattr(environment.os, "name", "nt")
    monkeypatch.setattr(environment.shutil, "which", lambda _name: "wsl.exe")
    monkeypatch.setattr(
        environment,
        "_query_nvidia_gpu",
        lambda _prefix=(): environment.NvidiaGPU("NVIDIA GeForce RTX 4070", 8.9, (580, 65, 6)),
    )

    def fake_run(command, **_kwargs):
        if command[-2:] == ["cat", "/etc/os-release"]:
            return SimpleNamespace(returncode=0, stdout='ID=ubuntu\nVERSION_ID="24.04"\n')
        if command[-2:] == ["uname", "-m"]:
            return SimpleNamespace(returncode=0, stdout="x86_64\n")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(environment.subprocess, "run", fake_run)

    plan, problem = environment.gromacs_install_plan()

    assert problem == ""
    assert plan is not None
    assert plan.gpu_backend == "CUDA"
    assert plan.source == "WSL"


def test_unsupported_linux_distribution_has_no_install_plan() -> None:
    plan, problem = environment._linux_gromacs_install_plan(
        apt_get="/usr/bin/apt-get",
        distro_id="fedora",
        distro_name="Fedora Linux",
        sudo="/usr/bin/sudo",
        is_root=False,
    )

    assert plan is None
    assert "Debian and Ubuntu" in problem


def test_install_runs_package_commands_then_verifies_version(monkeypatch) -> None:
    plan = GromacsInstallPlan(
        commands=(("sudo", "apt-get", "update"), ("sudo", "apt-get", "install", "-y", "gromacs")),
        probe_command=("gmx",),
        source="native",
        target="Ubuntu repositories (system-wide)",
    )
    installation = GromacsInstallation(
        command=("gmx",), version="2024.2", source="native", detail="GROMACS 2024.2 at /usr/bin/gmx."
    )
    probes = iter((None, installation))
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(environment, "probe_gromacs", lambda *_args: next(probes))
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda command, check: (commands.append(tuple(command)) or SimpleNamespace(returncode=0)),
    )

    result = environment.install_gromacs(plan)

    assert commands == list(plan.commands)
    assert "GROMACS 2024.2" in result
    assert "Ubuntu repositories" in result


def test_install_replaces_cpu_build_with_verified_cuda_build(monkeypatch) -> None:
    plan = GromacsInstallPlan(
        commands=(("sudo", "bash", "-lc", "build cuda gromacs"),),
        probe_command=("/usr/local/bin/gmx",),
        source="native",
        target="GROMACS 2026.4 built with CUDA",
        gpu_backend="CUDA",
    )
    cpu_installation = GromacsInstallation(
        command=("/usr/local/bin/gmx",),
        version="2024.2",
        source="native",
        detail="GROMACS 2024.2 at /usr/local/bin/gmx without GPU acceleration support.",
    )
    cuda_installation = GromacsInstallation(
        command=("/usr/local/bin/gmx",),
        version="2026.4",
        source="native",
        detail="GROMACS 2026.4 at /usr/local/bin/gmx with CUDA GPU support.",
        gpu_backend="CUDA",
    )
    probes = iter((cpu_installation, cuda_installation))
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(environment, "probe_gromacs", lambda *_args: next(probes))
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda command, check: (commands.append(tuple(command)) or SimpleNamespace(returncode=0)),
    )

    result = environment.install_gromacs(plan)

    assert commands == list(plan.commands)
    assert "CUDA GPU support" in result


def test_probe_reports_cuda_build_support(monkeypatch) -> None:
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="GROMACS version: 2026.4\nGPU support: CUDA\n",
            stderr="",
        ),
    )

    installation = environment.probe_gromacs(("gmx",))

    assert installation is not None
    assert installation.version == "2026.4"
    assert installation.gpu_backend == "CUDA"
    assert "CUDA GPU support" in installation.detail


def test_install_skips_package_changes_when_gromacs_is_already_available(monkeypatch) -> None:
    plan = GromacsInstallPlan(
        commands=(("sudo", "apt-get", "install", "-y", "gromacs"),),
        probe_command=("gmx",),
        source="native",
        target="Ubuntu repositories (system-wide)",
    )
    installation = GromacsInstallation(
        command=("gmx",), version="2024.2", source="native", detail="GROMACS 2024.2 at /usr/bin/gmx."
    )
    monkeypatch.setattr(environment, "probe_gromacs", lambda *_args: installation)
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("installer should not run")),
    )

    result = environment.install_gromacs(plan)

    assert "already available" in result
    assert "No package changes" in result
