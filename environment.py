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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


MINIMUM_PYTHON = (3, 10)
DEVELOPMENT_PACKAGES = {"pytest"}
OPTIONAL_PACKAGES = ("MDAnalysis", "biopython")
VERSION_PATTERN = re.compile(r"GROMACS version:\s*([^\s]+)", re.IGNORECASE)


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
    executable = command[-1] if source == "WSL" else command[0]
    detail = f"{executable} responds to --version."
    if version:
        detail = f"GROMACS {version} at {executable}."
    return GromacsInstallation(tuple(command), version, source, detail)


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
                "Install WSL2 with an Ubuntu distribution, then install GROMACS inside Ubuntu.",
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
                "Install an Ubuntu distribution, then install GROMACS inside it.",
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
