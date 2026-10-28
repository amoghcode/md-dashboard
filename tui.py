from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from environment import (
    EnvironmentCheck,
    EnvironmentReport,
    GromacsInstallation,
    inspect_environment,
    probe_gromacs,
    save_gromacs_command,
)
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Static, TabbedContent, TabPane


STATUS_STYLE = {
    "ready": "bold green",
    "missing": "bold red",
    "action": "bold yellow",
    "optional": "yellow",
    "info": "cyan",
}


def _status_text(check: EnvironmentCheck) -> Text:
    return Text(check.status.upper(), style=STATUS_STYLE.get(check.status, "white"))


def _render_checks(checks: tuple[EnvironmentCheck, ...], title: str) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", width=12)
    table.add_column("Details", ratio=3)
    for check in checks:
        table.add_row(check.name, _status_text(check), check.detail)
    return Panel(table, title=title, border_style="blue")


def build_environment_view(report: EnvironmentReport) -> Group:
    system = Table.grid(expand=True, padding=(0, 2))
    system.add_column(style="bold cyan", ratio=1)
    system.add_column(ratio=3)
    system.add_row("Operating system", report.operating_system)
    system.add_row("Python", report.python_check.detail)
    system.add_row("Python executable", report.python_executable)
    system.add_row("GROMACS", report.gromacs_check.detail)
    if report.installation:
        system.add_row("GROMACS command", shlex.join(report.installation.command))
        system.add_row("Discovery", report.installation.source)
    system.add_row("GMXLIB", report.gmxlib or "Not set")
    system.add_row("GMXDATA", report.gmxdata or "Not set")
    system.add_row("Saved settings", report.config_path)

    checks = [report.python_check, report.gromacs_check]
    if report.wsl_check:
        checks.append(report.wsl_check)

    sections = [
        Panel(system, title="Environment overview", border_style="cyan"),
        _render_checks(tuple(checks), "System checks"),
        _render_checks(report.required_packages, "Required Python packages"),
    ]
    if report.development_packages:
        sections.append(_render_checks(report.development_packages, "Development packages"))
    if report.optional_packages:
        sections.append(_render_checks(report.optional_packages, "Optional analysis packages"))
    return Group(*sections)


def build_home(report: EnvironmentReport) -> Group:
    required_missing = sum(check.status == "missing" for check in report.required_packages)
    environment_status = (
        "READY"
        if report.python_check.status == "ready" and required_missing == 0
        else "ACTION NEEDED"
    )
    environment_style = "green" if environment_status == "READY" else "yellow"
    gromacs_status = "READY" if report.installation else "NOT FOUND"
    gromacs_style = "green" if report.installation else "red"

    cards = Table.grid(expand=True, padding=(0, 1))
    cards.add_column(ratio=1)
    cards.add_column(ratio=1)
    cards.add_column(ratio=1)
    cards.add_row(
        Panel(f"[{environment_style}]{environment_status}[/{environment_style}]", title="Python environment"),
        Panel(f"[{gromacs_style}]{gromacs_status}[/{gromacs_style}]", title="GROMACS"),
        Panel(report.operating_system, title="Platform"),
    )

    return Group(
        Panel(
            Text("Molecular Dynamics Pipeline", style="bold white"),
            subtitle="Prototype · environment overview",
            border_style="bright_blue",
        ),
        cards,
    )


def _setup_instructions(report: EnvironmentReport) -> list[Panel]:
    panels: list[Panel] = []
    missing_python = [check.name for check in report.required_packages if check.status == "missing"]
    if report.python_check.status == "missing":
        panels.append(
            Panel(
                "Install Python 3.10 or newer, then launch the prototype with that interpreter.",
                title="Python",
                border_style="red",
            )
        )
    elif missing_python:
        if os.name == "nt":
            commands = (
                "py -3 -m venv .venv\n"
                ".venv\\Scripts\\Activate.ps1\n"
                "python -m pip install -r requirements.txt"
            )
        else:
            commands = (
                "python3 -m venv .venv\n"
                "source .venv/bin/activate\n"
                "python -m pip install -r requirements.txt"
            )
        panels.append(
            Panel(
                "The active Python environment is missing required packages. The setup action below can "
                "create a project .venv and install requirements.txt after confirmation.\n\n"
                f"Manual commands:\n{commands}",
                title="Python packages",
                border_style="yellow",
            )
        )
    else:
        panels.append(Panel("Python and required runtime packages are ready.", title="Python", border_style="green"))

    if not report.installation:
        if os.name == "nt":
            instructions = (
                "This prototype targets GROMACS through Ubuntu on WSL2. Install WSL2 and an Ubuntu distribution, "
                "then run the project from the Ubuntu terminal. Inside Ubuntu, install GROMACS with your "
                "distribution package manager, for example:\n\n"
                "sudo apt update\nsudo apt install gromacs python3-venv\n\n"
                "Rerun the environment check from Ubuntu to verify the installation."
            )
        elif shutil.which("apt-get"):
            instructions = (
                "GROMACS was not found. On this Debian/Ubuntu environment, install it from a system terminal:\n\n"
                "sudo apt update\nsudo apt install gromacs\n\n"
                "Rerun the environment check after installation."
            )
        else:
            instructions = (
                "GROMACS was not found. Install the package supported by your operating system, ensure its "
                "command is on PATH, and rerun the environment check."
            )
        panels.append(Panel(instructions, title="GROMACS", border_style="yellow"))
    else:
        panels.append(
            Panel(
                f"Detected {report.installation.detail}\n\n"
                "Use the GROMACS tab to verify and save a command for future runs.",
                title="GROMACS",
                border_style="green",
            )
        )

    panels.append(
        Panel(
            "The environment scan is read-only. Python package installation requires confirmation and is isolated "
            "in the project .venv. GROMACS installation is guided from the system terminal; shell startup files "
            "are never edited.",
            title="Setup behavior",
            border_style="dim",
        )
    )
    return panels


def _parse_command(value: str) -> tuple[str, ...]:
    raw = value.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]
    path = Path(raw).expanduser()
    if path.exists():
        return (str(path.resolve()),)
    try:
        return tuple(shlex.split(raw, posix=True))
    except ValueError:
        return ()


def _install_python_requirements() -> str:
    project_root = Path(__file__).resolve().parent
    venv_dir = project_root / ".venv"
    python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = project_root / "requirements.txt"

    if not python_path.exists():
        try:
            created = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            return f"Could not create the virtual environment: {error}"
        if created.returncode != 0 or not python_path.exists():
            detail = created.stderr.strip() or created.stdout.strip()
            return f"Virtual environment creation failed (exit {created.returncode}). {detail}".strip()

    try:
        installed = subprocess.run(
            [str(python_path), "-m", "pip", "install", "-r", str(requirements)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return f"Could not install Python requirements: {error}"
    if installed.returncode != 0:
        detail = installed.stderr.strip() or installed.stdout.strip()
        return f"Dependency installation failed (exit {installed.returncode}). {detail}".strip()

    activation = ".venv\\Scripts\\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
    return f"Python environment is ready. Activate it with: {activation}"


class ConfirmDialog(ModalScreen[bool]):
    """A small confirmation dialog for setup and configuration actions."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    ConfirmDialog {
        align: center middle;
    }
    #confirm-dialog {
        width: 72;
        height: auto;
        max-height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $panel;
    }
    #confirm-message {
        height: auto;
        margin-bottom: 1;
    }
    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="confirm-dialog"):
            yield Static(self.message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Continue", variant="primary", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


class MDPipelineApp(App[None]):
    """First navigable Textual prototype for the MD workflow."""

    TITLE = "Molecular Dynamics Pipeline"
    SUB_TITLE = "Environment setup and analysis prototype"
    BINDINGS = [("ctrl+q", "quit", "Quit"), ("ctrl+r", "refresh_environment", "Refresh checks")]

    CSS = """
    Screen {
        background: $surface;
    }
    TabPane {
        padding: 1 2;
    }
    VerticalScroll {
        height: 1fr;
    }
    #home-summary, #environment-view, #setup-copy, #analysis-intro {
        height: auto;
        margin-bottom: 1;
    }
    #setup-actions, #gromacs-actions, #monitor-actions {
        height: auto;
        margin: 1 0;
    }
    #analysis-structural-view {
        display: none;
        height: auto;
        margin-top: 1;
    }
    Button {
        margin-right: 1;
    }
    .section-title {
        color: $accent;
        text-style: bold;
        margin: 1 0;
    }
    .field-label {
        color: $text-muted;
        margin-top: 1;
    }
    #setup-status, #gromacs-status, #monitor-status {
        height: auto;
        color: $text-muted;
        margin: 1 0;
    }
    #monitor-view {
        height: 1fr;
        min-height: 12;
        border: round $primary;
        padding: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.report: EnvironmentReport | None = None
        self.gromacs_candidate: GromacsInstallation | None = None
        self.monitor: Any | None = None
        self.monitor_log_path: Path | None = None
        self.monitor_refresh_in_progress = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="home"):
            with TabPane("Overview", id="home"):
                with VerticalScroll():
                    yield Static("Scanning this machine's Python and GROMACS environment…", id="home-summary")
            with TabPane("Environment", id="environment"):
                with VerticalScroll():
                    yield Static("Environment scan running…", id="environment-view")
                    yield Button("Refresh checks", id="refresh-environment", variant="primary")
            with TabPane("Setup", id="setup"):
                with VerticalScroll():
                    yield Static("Environment scan running…", id="setup-copy")
                    with Horizontal(id="setup-actions"):
                        yield Button("Create .venv and install Python packages", id="setup-python", variant="primary")
                    yield Static("", id="setup-status")
            with TabPane("GROMACS", id="gromacs"):
                with VerticalScroll():
                    yield Static("Verify an installed GROMACS executable or command, then save it to user settings.")
                    yield Static("Command", classes="field-label")
                    yield Input(value="gmx", placeholder="gmx, gmx_mpi, or a full executable path", id="gromacs-command")
                    with Horizontal(id="gromacs-actions"):
                        yield Button("Verify command", id="verify-gromacs", variant="primary")
                        yield Button("Save verified command", id="save-gromacs", disabled=True)
                    yield Static("", id="gromacs-status")
            with TabPane("Monitor", id="monitor"):
                with VerticalScroll():
                    yield Static("Watch an existing GROMACS log. This prototype does not launch a simulation.")
                    yield Static("GROMACS log path", classes="field-label")
                    yield Input(placeholder="/path/to/md.log", id="monitor-log-path")
                    yield Static("Optional MDP file", classes="field-label")
                    yield Input(placeholder="/path/to/run.mdp", id="monitor-mdp-path")
                    with Horizontal(id="monitor-actions"):
                        yield Button("Start monitoring", id="start-monitor", variant="primary")
                    yield Static("Enter a log path to begin.", id="monitor-status")
                    yield Static("", id="monitor-view")
            with TabPane("Analysis", id="analysis"):
                with VerticalScroll():
                    yield Static(
                        "Select Structural to reveal its placeholder options. Thermodynamic is reserved for a future update.",
                        id="analysis-intro",
                    )
                    with Horizontal(id="analysis-categories"):
                        yield Button("Structural", id="analysis-structural", variant="primary")
                        yield Button("Thermodynamic", id="analysis-thermodynamic")
                    with Vertical(id="analysis-structural-view"):
                        yield Static(
                            "These options are placeholders; selecting one currently has no effect."
                        )
                        with Horizontal(id="analysis-methods"):
                            yield Button("RMSD", id="analysis-rmsd")
                            yield Button("RMSF", id="analysis-rmsf")
                            yield Button("RoG", id="analysis-rog")
        yield Footer()

    def on_mount(self) -> None:
        self._scan_environment()
        self.set_interval(2.0, self._refresh_monitor)

    @work(thread=True, exclusive=True)
    def _scan_environment(self) -> None:
        report = inspect_environment()
        self.call_from_thread(self._apply_environment_report, report)

    def _apply_environment_report(self, report: EnvironmentReport) -> None:
        self.report = report
        self.query_one("#home-summary", Static).update(build_home(report))
        self.query_one("#environment-view", Static).update(build_environment_view(report))
        self.query_one("#setup-copy", Static).update(Group(*_setup_instructions(report)))
        command_input = self.query_one("#gromacs-command", Input)
        if command_input.value == "gmx" and report.installation:
            command_input.value = shlex.join(report.installation.command)

    def action_refresh_environment(self) -> None:
        self.query_one("#environment-view", Static).update("Refreshing environment checks…")
        self._scan_environment()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "analysis-structural":
            self.query_one("#analysis-structural-view", Vertical).display = True
        elif button_id == "refresh-environment":
            self.action_refresh_environment()
        elif button_id == "setup-python":
            self.push_screen(
                ConfirmDialog(
                    "This will create or reuse a .venv folder in the project and install every package in "
                    "requirements.txt into it. Packages are installed only inside that virtual environment. Continue?"
                ),
                self._confirm_python_setup,
            )
        elif button_id == "verify-gromacs":
            self._begin_gromacs_verification()
        elif button_id == "save-gromacs":
            self._confirm_save_gromacs()
        elif button_id == "start-monitor":
            self._start_monitor()

    def _confirm_python_setup(self, confirmed: bool | None) -> None:
        if confirmed:
            self.query_one("#setup-status", Static).update("Setting up the project Python environment…")
            self._setup_python_worker()

    @work(thread=True, exclusive=True)
    def _setup_python_worker(self) -> None:
        result = _install_python_requirements()
        self.call_from_thread(self._show_setup_result, result)

    def _show_setup_result(self, message: str) -> None:
        self.query_one("#setup-status", Static).update(message)
        if message.startswith("Python environment is ready"):
            self._scan_environment()

    def _begin_gromacs_verification(self) -> None:
        value = self.query_one("#gromacs-command", Input).value
        command = _parse_command(value)
        status = self.query_one("#gromacs-status", Static)
        save_button = self.query_one("#save-gromacs", Button)
        save_button.disabled = True
        self.gromacs_candidate = None
        if not command:
            status.update("Enter an executable path or command, such as gmx or gmx_mpi.")
            return
        status.update("Checking the command's --version response…")
        self._verify_gromacs_worker(command)

    @work(thread=True, exclusive=True)
    def _verify_gromacs_worker(self, command: tuple[str, ...]) -> None:
        installation = probe_gromacs(command)
        self.call_from_thread(self._show_gromacs_verification, installation)

    def _show_gromacs_verification(self, installation: GromacsInstallation | None) -> None:
        self.gromacs_candidate = installation
        save_button = self.query_one("#save-gromacs", Button)
        if installation:
            self.query_one("#gromacs-status", Static).update(f"Verified: {installation.detail}")
            save_button.disabled = False
        else:
            self.query_one("#gromacs-status", Static).update(
                "Verification failed. The command was not saved; check that it is installed and on PATH."
            )
            save_button.disabled = True

    def _confirm_save_gromacs(self) -> None:
        if not self.gromacs_candidate:
            return
        self.push_screen(
            ConfirmDialog("Save this verified GROMACS command to your per-user settings file?"),
            self._save_gromacs_if_confirmed,
        )

    def _save_gromacs_if_confirmed(self, confirmed: bool | None) -> None:
        if not confirmed or not self.gromacs_candidate:
            return
        installation = self.gromacs_candidate
        command = installation.command
        path = save_gromacs_command(command, installation.version)
        self.query_one("#gromacs-status", Static).update(f"Saved verified command to {path}")
        self._scan_environment()

    def _start_monitor(self) -> None:
        raw_log = self.query_one("#monitor-log-path", Input).value.strip().strip('"')
        raw_mdp = self.query_one("#monitor-mdp-path", Input).value.strip().strip('"')
        if not raw_log:
            self.query_one("#monitor-status", Static).update("Enter a GROMACS log path first.")
            return
        from monitor import SimulationMonitor

        self.monitor_log_path = Path(raw_log).expanduser()
        mdp_path = Path(raw_mdp).expanduser() if raw_mdp else None
        self.monitor = SimulationMonitor(self.monitor_log_path, mdp_path)
        self.query_one("#monitor-status", Static).update(f"Monitoring {self.monitor_log_path}")
        self._refresh_monitor()

    def _refresh_monitor(self) -> None:
        if (
            self.monitor is None
            or self.monitor_log_path is None
            or self.monitor_refresh_in_progress
        ):
            return
        self.monitor_refresh_in_progress = True
        self._monitor_worker(self.monitor, self.monitor_log_path)

    @work(thread=True, exclusive=True)
    def _monitor_worker(self, monitor: Any, log_path: Path) -> None:
        from dashboard import build_dashboard

        try:
            snapshot = monitor.snapshot()
            renderable = build_dashboard(snapshot, log_path)
        except Exception as error:
            self.call_from_thread(self._show_monitor_error, str(error))
            return
        self.call_from_thread(self._show_monitor_renderable, renderable)

    def _show_monitor_renderable(self, renderable: Any) -> None:
        self.query_one("#monitor-view", Static).update(renderable)
        self.monitor_refresh_in_progress = False

    def _show_monitor_error(self, message: str) -> None:
        self.query_one("#monitor-status", Static).update(f"Monitor refresh failed: {message}")
        self.monitor_refresh_in_progress = False


def launch_tui() -> None:
    MDPipelineApp().run()


if __name__ == "__main__":
    launch_tui()
