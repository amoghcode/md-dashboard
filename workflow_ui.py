"""Background analysis and prepared simulation workflows."""
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from textual import work
from textual.screen import Screen
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static
from execution import gmx_command

class WorkflowScreen(Screen):
    def __init__(self, mode, command):
        super().__init__()
        self.mode, self.command, self.process = mode, command, None
    def compose(self):
        with VerticalScroll():
            yield Static(self.mode + " — use matching inputs; run inside WSL for Linux GROMACS.")
            if self.mode == "Simulation":
                yield Static("Prepared input folder containing box_mixed.gro, system.top, force field and MDP files")
                yield Input(id="source")
                yield Static("Optional existing run inputs folder to resume production (leave blank for new run)")
                yield Input(id="resume")
                yield Static("Steps per stage: 5000 for a smoke test; blank preserves full protocol")
                yield Input(value="5000", id="steps")
                yield Static("Execution: auto, cpu or gpu")
                yield Input(value="auto", id="device")
            else:
                yield Static("Topology (.tpr)")
                yield Input(id="topology")
                yield Static("Trajectory (.xtc/.trr), matching topology atom order")
                yield Input(id="trajectory")
                yield Static("Optional index (.ndx)")
                yield Input(id="index")
                yield Static("Group (must exist in the selected system)")
                yield Input(value="Backbone", id="group")
                yield Static("Start and end time in ns (-1 means all remaining frames)")
                yield Input(value="0", id="begin")
                yield Input(value="-1", id="end")
            yield Static("New results parent folder")
            yield Input(value=str(Path("results").resolve()), id="destination")
            yield Button("Run", id="execute")
            yield Button("Stop", id="stop", disabled=True)
            yield Button("Back", id="back")
            yield Static("", id="run-stage", markup=False)
            yield Static("", id="live-monitor")
            yield Static("", id="state", markup=False)
    def on_button_pressed(self, event):
        if event.button.id == "back":
            if self.process and self.process.poll() is None:
                self.query_one("#state", Static).update("Stop the active job before leaving.")
            else:
                self.app.pop_screen()
        elif event.button.id == "stop" and self.process and self.process.poll() is None:
            if os.name == "posix":
                import signal
                os.killpg(self.process.pid, signal.SIGTERM)
            else:
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True)
        elif event.button.id == "execute":
            self.query_one("#execute", Button).disabled = True
            values = {w.id:w.value for w in self.query(Input)}
            self.run_job(values)
    def show_run_monitor(self, stage, renderable):
        self.query_one("#run-stage", Static).update("Active stage: " + stage)
        self.query_one("#live-monitor", Static).update(renderable)

    def set_running(self, running):
        self.query_one("#stop", Button).disabled = not running
        self.query_one("#execute", Button).disabled = running
        if running and self.mode == "Simulation":
            self.query_one("#run-stage", Static).update("Preparing simulation; waiting for the first stage...")
            self.query_one("#run-stage", Static).scroll_visible()


    @work(thread=True)
    def run_job(self, values):
        folder = None
        try:
            env = dict(os.environ, MD_GMX_COMMAND=self.command)
            old = os.environ.get("MD_GMX_COMMAND")
            # Validate without changing process-wide environment.
            import shlex
            tokens = shlex.split(self.command)
            if not tokens:
                raise ValueError("Choose a GROMACS command.")
            is_wsl = "wsl" in Path(tokens[0]).name.lower()
            folder = Path(values["destination"]).expanduser().resolve() / (self.mode.lower()+"-"+uuid.uuid4().hex[:10])
            folder.mkdir(parents=True)
            for key in ("topology", "trajectory", "index", "source", "resume"):
                if values.get(key):
                    values[key] = str(Path(values[key]).expanduser().resolve())
            request = dict(values, mode=self.mode, command=tokens)
            (folder/"request.json").write_text(json.dumps(request, indent=2))
            request_path = folder/"request.json"
            worker_command = [sys.executable, str(Path(__file__).with_name("workflow_worker.py")), str(request_path)]
            if is_wsl:
                from wsl_execution import worker_command as wsl_worker_command
                worker_command = wsl_worker_command(tokens, request_path, env)
            follower = None
            if self.mode == "Simulation":
                from run_monitor import LaunchedRunMonitor
                follower = LaunchedRunMonitor(folder, values.get("resume") or folder/"inputs")
            self.app.call_from_thread(self.set_running, True)
            def refresh():
                if follower:
                    try:
                        state = follower.snapshot()
                        if state:
                            from dashboard import build_dashboard
                            stage, snapshot = state
                            view = build_dashboard(snapshot, follower.inputs_folder/(stage+".log"))
                            self.app.call_from_thread(self.show_run_monitor, stage, view)
                    except Exception as error:
                        self.app.call_from_thread(self.query_one("#run-stage", Static).update,
                                                  "Monitor refresh: "+str(error))
            with (folder/"execution.log").open("w") as log:
                self.process = subprocess.Popen(worker_command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=(os.name == "posix"))
                import time
                while self.process.poll() is None:
                    refresh()
                    text = (folder/"execution.log").read_text(errors="replace")
                    self.app.call_from_thread(self.query_one("#state", Static).update, "Running: "+str(folder)+"\n"+text[-2500:])
                    time.sleep(.5)
            refresh()
            if self.process.returncode:
                raise RuntimeError((folder/"execution.log").read_text(errors="replace")[-3000:])
            self.app.call_from_thread(self.query_one("#state", Static).update, "Completed. Results: "+str(folder))
        except Exception as exc:
            self.app.call_from_thread(self.query_one("#state", Static).update, "Stopped or failed: "+str(exc))
        finally:
            self.app.call_from_thread(self.set_running, False)
