from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence


MINIMUM_PYTHON = (3, 10)
DEVELOPMENT_PACKAGES = {"pytest"}
OPTIONAL_PACKAGES = ("MDAnalysis", "biopython")
VERSION_PATTERN = re.compile(r"GROMACS version:\s*([^\s]+)", re.IGNORECASE)
GPU_SUPPORT_PATTERN = re.compile(r"GPU support:\s*([^\r\n]+)", re.IGNORECASE)
NVIDIA_GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=name,compute_cap,driver_version",
    "--format=csv,noheader",
)
GROMACS_GPU_BUILD_VERSION = "2026.4"
GROMACS_GPU_BUILD_MD5 = "fbe7d5071dee17e3f448502b1fd30d55"
GROMACS_GPU_BUILD_URL = (
    f"https://ftp.gromacs.org/gromacs/gromacs-{GROMACS_GPU_BUILD_VERSION}.tar.gz"
)


@dataclass(frozen=True)
class EnvironmentCheck:
    name: str
    status: str
    detail: str
    recommendation: str | None = None


@dataclass(frozen=True)
class GromacsInstallation:
    command: tuple[str, ...]
    version: str | None
    source: str
    detail: str
    gpu_backend: str | None = None


@dataclass(frozen=True)
class GromacsInstallPlan:
    commands: tuple[tuple[str, ...], ...]
    probe_command: tuple[str, ...]
    source: str
    target: str
    gpu_backend: str | None = None
    note: str = ""


@dataclass(frozen=True)
class NvidiaGPU:
    name: str
    compute_capability: float
    driver_version: tuple[int, ...]


@dataclass(frozen=True)
class EnvironmentReport:
    operating_system: str
    python_version: str
    python_executable: str
    python_check: EnvironmentCheck
    required_packages: tuple[EnvironmentCheck, ...]
    development_packages: tuple[EnvironmentCheck, ...]
    optional_packages: tuple[EnvironmentCheck, ...]
    gromacs_check: EnvironmentCheck
    installation: GromacsInstallation | None
    configured_command: tuple[str, ...] | None
    config_path: str
    gmxlib: str | None
    gmxdata: str | None
    wsl_check: EnvironmentCheck | None


def _linux_gromacs_install_plan(
    *,
    apt_get: str | None,
    distro_id: str,
    distro_name: str,
    sudo: str | None,
    is_root: bool,
) -> tuple[GromacsInstallPlan | None, str]:
    if distro_id.lower() not in {"debian", "ubuntu"} or not apt_get:
        return None, "Automatic GROMACS installation is supported on Debian and Ubuntu with apt-get."
    if not is_root and not sudo:
        return None, "Install GROMACS from a system terminal with administrator privileges."

    privilege = () if is_root else (sudo,)
    commands = (
        (*privilege, apt_get, "update"),
        (*privilege, apt_get, "install", "-y", "gromacs"),
    )
    target = f"{distro_name or distro_id} package repositories (system-wide)"
    return GromacsInstallPlan(commands, ("gmx",), "native", target), ""


def _parse_nvidia_gpu_query(output: str) -> NvidiaGPU | None:
    """Return the oldest GROMACS-compatible GPU to support mixed-GPU machines."""

    compatible: list[NvidiaGPU] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            capability = float(fields[1])
        except ValueError:
            continue
        driver_version = tuple(int(part) for part in re.findall(r"\d+", fields[2]))
        if not driver_version:
            continue
        if capability >= 5.0:
            compatible.append(NvidiaGPU(fields[0], capability, driver_version))
    return min(compatible, key=lambda device: device.compute_capability) if compatible else None


def _gpu_build_script(
    cuda_repository: str,
    toolkit_package: str,
    toolkit_version: str,
) -> str:
    """Build a pinned GROMACS release with CUDA using the official source archive."""

    return f'''set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates wget build-essential cmake
wget -q -O /tmp/cuda-keyring.deb "{cuda_repository}/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y {toolkit_package}

work_dir="$(mktemp -d /tmp/md-gromacs-cuda-build.XXXXXX)"
trap 'case "$work_dir" in /tmp/md-gromacs-cuda-build.*) rm -rf -- "$work_dir" ;; esac' EXIT
cd "$work_dir"
wget -q -O gromacs.tar.gz "{GROMACS_GPU_BUILD_URL}"
echo "{GROMACS_GPU_BUILD_MD5}  gromacs.tar.gz" | md5sum --check -
tar -xzf gromacs.tar.gz

for nvcc in /usr/local/cuda-{toolkit_version}/bin/nvcc /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    if [ -x "$nvcc" ]; then
        export PATH="$(dirname "$nvcc"):$PATH"
        export CUDAToolkit_ROOT="$(dirname "$(dirname "$nvcc")")"
        break
    fi
done
command -v nvcc >/dev/null
cmake -S "gromacs-{GROMACS_GPU_BUILD_VERSION}" -B build \\
    -DGMX_GPU=CUDA -DGMX_BUILD_OWN_FFTW=ON -DCMAKE_INSTALL_PREFIX=/usr/local
jobs="$(nproc)"
if [ "$jobs" -gt 4 ]; then jobs=4; fi
cmake --build build --parallel "$jobs"
cmake --install build
ldconfig
/usr/local/bin/gmx --version
'''


def _nvidia_cuda_gromacs_install_plan(
    *,
    source: str,
    runner: Sequence[str],
    distro_id: str,
    distro_version: str,
    architecture: str,
    gpu: NvidiaGPU,
) -> tuple[GromacsInstallPlan | None, str]:
    """Build a GPU plan only for a known supported Ubuntu x86-64 environment."""

    normalized_arch = architecture.lower()
    if normalized_arch not in {"x86_64", "amd64"}:
        return None, "Automatic CUDA builds currently require a 64-bit x86 Ubuntu environment."
    if distro_id.lower() != "ubuntu" or distro_version not in {"24.04", "26.04"}:
        return None, "Automatic CUDA builds currently support Ubuntu 24.04 and 26.04."
    if gpu.compute_capability < 5.0:
        return None, "This NVIDIA GPU is below GROMACS's CUDA compute-capability 5.0 minimum."

    repository_version = distro_version.replace(".", "")
    repository = (
        f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{repository_version}/x86_64"
    )
    # CUDA 13 dropped code generation for Maxwell, Pascal, and Volta. CUDA
    # 12.9 remains the compatible choice for those GPUs and older drivers.
    if repository_version == "2604":
        toolkit_package = "cuda-toolkit"
        toolkit_version = "13.4"
        minimum_driver = (615, 0)
        if gpu.compute_capability < 7.5:
            return None, "This older GPU needs CUDA 12.9; automatic CUDA 12.9 builds require Ubuntu 24.04."
    elif gpu.compute_capability >= 7.5 and gpu.driver_version >= (580,):
        toolkit_package = "cuda-toolkit-13-0"
        toolkit_version = "13.0"
        minimum_driver = (580,)
    else:
        toolkit_package = "cuda-toolkit-12-9"
        toolkit_version = "12.9"
        minimum_driver = (528, 33) if source == "WSL" else (525, 60, 13)
    if gpu.driver_version < minimum_driver:
        required = ".".join(str(part) for part in minimum_driver)
        current = ".".join(str(part) for part in gpu.driver_version)
        host = "Windows host" if source == "WSL" else "Linux system"
        return None, (
            f"The NVIDIA driver ({current}) is too old for this CUDA build. Update the driver on the {host} "
            f"to version {required} or newer, then refresh checks."
        )
    if toolkit_version == "12.9" and repository_version != "2404":
        return None, "CUDA 12.9 builds for older GPUs or drivers currently require Ubuntu 24.04."
    script = _gpu_build_script(repository, toolkit_package, toolkit_version)
    command = (*runner, "bash", "-lc", script)
    target = (
        f"GROMACS {GROMACS_GPU_BUILD_VERSION} built with CUDA {toolkit_version} for {gpu.name} "
        "(system-wide; NVIDIA CUDA Toolkit from NVIDIA's signed APT repository)"
    )
    return (
        GromacsInstallPlan(
            commands=(command,),
            probe_command=("/usr/local/bin/gmx",) if source == "native" else (*runner, "/usr/local/bin/gmx"),
            source=source,
            target=target,
            gpu_backend="CUDA",
        ),
        "",
    )


def _wsl_gromacs_install_plan(
    wsl: str,
    os_release: str,
    *,
    note: str = "",
) -> tuple[GromacsInstallPlan | None, str]:
    distro_id = ""
    for line in os_release.splitlines():
        if line.startswith("ID="):
            distro_id = line.partition("=")[2].strip().strip('"').lower()
            break
    if distro_id not in {"debian", "ubuntu"}:
        return None, "Automatic Windows installation needs an existing default Ubuntu or Debian WSL distribution."

    root_apt = (wsl, "--user", "root", "--exec", "apt-get")
    commands = (
        (*root_apt, "update"),
        (*root_apt, "install", "-y", "gromacs"),
    )
    target = f"the default WSL {distro_id.title()} package repositories (system-wide)"
    probe_command = (wsl, "--exec", "gmx")
    return GromacsInstallPlan(commands, probe_command, "WSL", target, note=note), ""


def _parse_os_release(os_release: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in os_release.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value.strip().strip('"').strip("'")
    return values


def _query_nvidia_gpu(command_prefix: Sequence[str] = ()) -> NvidiaGPU | None:
    commands = [[*command_prefix, *NVIDIA_GPU_QUERY]]
    if command_prefix and "wsl" in Path(command_prefix[0]).name.lower():
        commands.append(
            [*command_prefix, "/usr/lib/wsl/lib/nvidia-smi", *NVIDIA_GPU_QUERY[1:]]
        )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            gpu = _parse_nvidia_gpu_query(result.stdout)
            if gpu:
                return gpu
    return None


def gromacs_install_plan() -> tuple[GromacsInstallPlan | None, str]:
    """Return a supported system GROMACS plan, preferring CUDA for compatible NVIDIA GPUs."""

    if os.name == "nt":
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if not wsl:
            return None, "Install WSL2 with Ubuntu 24.04 to enable automatic CUDA builds from this app."
        try:
            result = subprocess.run(
                [wsl, "--exec", "cat", "/etc/os-release"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, "Could not inspect the default WSL distribution. Start Ubuntu and refresh the checks."
        if result.returncode != 0:
            return None, (
                "No working default WSL distribution was found. Install Ubuntu 24.04, make it the default "
                "distribution, then refresh the checks."
            )
        release = _parse_os_release(result.stdout)
        gpu = _query_nvidia_gpu((wsl, "--exec"))
        if gpu:
            try:
                architecture_result = subprocess.run(
                    [wsl, "--exec", "uname", "-m"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
                architecture = (
                    architecture_result.stdout.strip() if architecture_result.returncode == 0 else ""
                )
            except (OSError, subprocess.TimeoutExpired):
                architecture = ""
            plan, problem = _nvidia_cuda_gromacs_install_plan(
                source="WSL",
                runner=(wsl, "--user", "root", "--exec"),
                distro_id=release.get("ID", ""),
                distro_version=release.get("VERSION_ID", ""),
                architecture=architecture,
                gpu=gpu,
            )
            if plan:
                return plan, ""
            cpu_plan, cpu_problem = _wsl_gromacs_install_plan(wsl, result.stdout, note=problem)
            if cpu_plan:
                return cpu_plan, ""
            return cpu_plan, cpu_problem or problem
        cpu_plan, cpu_problem = _wsl_gromacs_install_plan(
            wsl,
            result.stdout,
            note=(
                "No compatible NVIDIA GPU was detected inside WSL. The repository package may use the CPU only. "
                "Automatic GPU builds currently target NVIDIA GPUs with CUDA."
            ),
        )
        return cpu_plan, cpu_problem

    if platform.system() != "Linux":
        return None, "Automatic GROMACS installation currently supports Debian and Ubuntu only."

    try:
        release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        release = {}
    apt_get = shutil.which("apt-get")
    sudo = shutil.which("sudo")
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    gpu = _query_nvidia_gpu()
    if gpu and (is_root or sudo):
        plan, problem = _nvidia_cuda_gromacs_install_plan(
            source="native",
            runner=(sudo,) if not is_root and sudo else (),
            distro_id=release.get("ID", "").strip('"'),
            distro_version=release.get("VERSION_ID", "").strip('"'),
            architecture=platform.machine(),
            gpu=gpu,
        )
        if plan:
            return plan, ""
        cpu_plan, cpu_problem = _linux_gromacs_install_plan(
            apt_get=apt_get,
            distro_id=release.get("ID", "").strip('"'),
            distro_name=release.get("PRETTY_NAME", release.get("NAME", "")),
            sudo=sudo,
            is_root=is_root,
        )
        if cpu_plan:
            return replace(cpu_plan, note=problem), ""
        return cpu_plan, cpu_problem or problem
    cpu_plan, cpu_problem = _linux_gromacs_install_plan(
        apt_get=apt_get,
        distro_id=release.get("ID", "").strip('"'),
        distro_name=release.get("PRETTY_NAME", release.get("NAME", "")),
        sudo=sudo,
        is_root=is_root,
    )
    if cpu_plan:
        return replace(
            cpu_plan,
            note=(
                "No compatible NVIDIA GPU was detected. The repository package may use the CPU only. "
                "Automatic GPU builds currently target NVIDIA GPUs with CUDA."
            ),
        ), ""
    return cpu_plan, cpu_problem


def install_gromacs(plan: GromacsInstallPlan) -> str:
    """Install/build GROMACS with the confirmed system plan, then verify its features."""

    existing = probe_gromacs(plan.probe_command, plan.source)
    if existing and (not plan.gpu_backend or existing.gpu_backend == plan.gpu_backend):
        return f"GROMACS is already available: {existing.detail} No package changes were made."

    for command in plan.commands:
        try:
            result = subprocess.run(command, check=False)
        except (OSError, KeyboardInterrupt) as error:
            if isinstance(error, KeyboardInterrupt):
                return "GROMACS installation was interrupted."
            return f"Could not start the GROMACS package installer: {error}"
        if result.returncode != 0:
            return (
                f"GROMACS installation/build failed (exit code {result.returncode}). "
                "Review the package-manager or build output above."
            )

    installation = probe_gromacs(plan.probe_command, plan.source)
    if not installation:
        return "The package manager finished, but `gmx --version` did not verify. Refresh checks or review the package-manager output."
    if plan.gpu_backend and installation.gpu_backend != plan.gpu_backend:
        return (
            f"GROMACS installed, but the requested {plan.gpu_backend} GPU support was not found in its build. "
            "Review the build output and refresh the environment checks."
        )
    return f"GROMACS installed and verified: {installation.detail} Source: {plan.target}."


def settings_path() -> Path:
    """Return the per-user settings path without creating it."""

    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "md-dashboard" / "settings.json"


def load_settings() -> dict[str, Any]:
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_gromacs_command(command: Sequence[str], version: str | None) -> Path:
    """Persist a verified command after the user explicitly chooses to save it."""

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_settings()
    data["gromacs_command"] = list(command)
    data["gromacs_version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _requirement_names() -> tuple[str, ...]:
    requirements = Path(__file__).with_name("requirements.txt")
    if not requirements.exists():
        return ("rich",)

    names: list[str] = []
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", ".")):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            name = match.group(1).replace("_", "-").lower()
            if name not in names:
                names.append(name)
    return tuple(names or ("rich",))


def _package_check(name: str, *, optional: bool = False) -> EnvironmentCheck:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if optional:
            return EnvironmentCheck(name, "optional", "Not installed; this package is optional.")
        return EnvironmentCheck(
            name,
            "missing",
            "Not installed in the active Python environment.",
            "Install the project requirements in the environment used to launch the app.",
        )
    return EnvironmentCheck(
        name,
        "ready" if not optional else "optional",
        f"Installed{f' ({version})' if version else ''}." + (" Optional package." if optional else ""),
    )


def probe_gromacs(command: Sequence[str], source: str = "configured") -> GromacsInstallation | None:
    try:
        result = subprocess.run(
            [*command, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        return None
    match = VERSION_PATTERN.search(output)
    version = match.group(1) if match else None
    gpu_match = GPU_SUPPORT_PATTERN.search(output)
    gpu_support = gpu_match.group(1).strip() if gpu_match else ""
    gpu_backend_name = gpu_support.split(maxsplit=1)[0].strip("(),:") if gpu_support else ""
    gpu_backend = (
        gpu_backend_name
        if gpu_backend_name and gpu_backend_name.lower() not in {"none", "disabled", "no"}
        else None
    )
    executable = command[-1] if source == "WSL" else command[0]
    detail = f"{executable} responds to --version."
    if version:
        detail = f"GROMACS {version} at {executable}."
    if gpu_backend:
        detail = detail[:-1] + f" with {gpu_backend} GPU support."
    else:
        detail = detail[:-1] + " without GPU acceleration support."
    return GromacsInstallation(tuple(command), version, source, detail, gpu_backend)


def _detect_gromacs(saved_command: Sequence[str] | None) -> tuple[
    GromacsInstallation | None, EnvironmentCheck | None
]:
    if saved_command:
        saved = probe_gromacs(saved_command, "configured")
        if saved:
            return saved, None

    for name in ("gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d"):
        executable = shutil.which(name)
        if executable:
            installation = probe_gromacs((executable,), "native")
            if installation:
                return installation, None

    wsl_check: EnvironmentCheck | None = None
    if os.name == "nt":
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if not wsl:
            wsl_check = EnvironmentCheck(
                "WSL2",
                "missing",
                "Windows Subsystem for Linux was not found.",
                "Install WSL2 with Ubuntu 24.04 to enable automatic CUDA builds, then install GROMACS inside Ubuntu.",
            )
            return None, wsl_check

        try:
            distributions = subprocess.run(
                [wsl, "--list", "--quiet"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            distributions = None
        names = (
            [line.replace("\x00", "").strip() for line in distributions.stdout.splitlines()]
            if distributions and distributions.returncode == 0
            else []
        )
        names = [name for name in names if name and not set(name) <= {"*", "-", " "}]
        if not names:
            wsl_check = EnvironmentCheck(
                "WSL2",
                "action",
                "WSL is available, but no Linux distribution is registered.",
                "Install Ubuntu 24.04 to enable automatic CUDA builds, then install GROMACS inside it.",
            )
            return None, wsl_check

        installation = probe_gromacs((wsl, "--exec", "gmx"), "WSL")
        if installation:
            wsl_check = EnvironmentCheck("WSL2", "ready", "GROMACS is available in the default WSL distribution.")
            return installation, wsl_check
        wsl_check = EnvironmentCheck(
            "WSL2",
            "action",
            "WSL is installed, but GROMACS was not found in its default distribution.",
            "Install GROMACS in the WSL distribution and rerun this check.",
        )

    return None, wsl_check


def inspect_environment() -> EnvironmentReport:
    settings = load_settings()
    raw_command = settings.get("gromacs_command")
    saved_command = (
        tuple(str(item) for item in raw_command)
        if isinstance(raw_command, list) and raw_command and all(raw_command)
        else None
    )

    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    python_check = EnvironmentCheck(
        "Python",
        "ready" if python_ok else "missing",
        f"Python {platform.python_version()} at {sys.executable}.",
        None if python_ok else "Python 3.10 or newer is required by this codebase.",
    )

    package_names = _requirement_names()
    required = tuple(
        _package_check(name) for name in package_names if name not in DEVELOPMENT_PACKAGES
    )
    development = tuple(
        _package_check(name) for name in package_names if name in DEVELOPMENT_PACKAGES
    )
    optional = tuple(_package_check(name, optional=True) for name in OPTIONAL_PACKAGES)

    installation, wsl_check = _detect_gromacs(saved_command)
    if installation:
        gromacs_check = EnvironmentCheck("GROMACS", "ready", installation.detail)
    else:
        gromacs_check = EnvironmentCheck(
            "GROMACS",
            "missing",
            "No working GROMACS command was found on PATH or in the default WSL distribution.",
            "Use the setup guide to install GROMACS, then rerun the environment check.",
        )

    configured = installation.command if installation and installation.source == "configured" else None
    return EnvironmentReport(
        operating_system=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        python_executable=sys.executable,
        python_check=python_check,
        required_packages=required,
        development_packages=development,
        optional_packages=optional,
        gromacs_check=gromacs_check,
        installation=installation,
        configured_command=configured,
        config_path=str(settings_path()),
        gmxlib=os.environ.get("GMXLIB"),
        gmxdata=os.environ.get("GMXDATA"),
        wsl_check=wsl_check,
    )


def report_as_dict(report: EnvironmentReport) -> dict[str, Any]:
    data = asdict(report)
    data["config_path"] = str(data["config_path"])
    return data


def main() -> None:
    """Allow dependency and GROMACS checks before Rich is installed."""

    parser = argparse.ArgumentParser(description="Inspect the MD Python and GROMACS environment.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    report = inspect_environment()
    if args.json:
        print(json.dumps(report_as_dict(report), indent=2, default=str))
        return

    print(f"Operating system: {report.operating_system}")
    print(f"Python: {report.python_check.status.upper()} — {report.python_check.detail}")
    for check in (*report.required_packages, *report.development_packages, *report.optional_packages):
        print(f"{check.name}: {check.status.upper()} — {check.detail}")
    print(f"GROMACS: {report.gromacs_check.status.upper()} — {report.gromacs_check.detail}")
    print(f"Settings: {report.config_path}")


if __name__ == "__main__":
    main()
