"""CLI helpers to manage the NiceGUI UI process."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _state_root() -> Path:
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "steve-recommender"
    return Path.home() / ".local" / "state" / "steve-recommender"


def _pid_path(state_dir: Path) -> Path:
    return state_dir / "steve-ui.pid"


def _log_path(state_dir: Path) -> Path:
    return state_dir / "steve-ui.log"


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _wait_for_exit(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_running(pid):
            return True
        time.sleep(0.2)
    return False


def _start(state_dir: Path) -> int:
    pid_file = _pid_path(state_dir)
    existing_pid = _read_pid(pid_file)
    if existing_pid and _is_running(existing_pid):
        print(f"UI already running (pid {existing_pid}).")
        return 0
    if existing_pid and not _is_running(existing_pid):
        pid_file.unlink(missing_ok=True)

    state_dir.mkdir(parents=True, exist_ok=True)
    log_file = _log_path(state_dir).open("a", encoding="utf-8")
    cmd = [sys.executable, "-m", "steve_recommender.ui_nicegui.app"]
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    print(f"Started UI (pid {proc.pid}).")
    print(f"Log: {_log_path(state_dir)}")
    print("URL: http://localhost:8080")
    return 0


def _stop(state_dir: Path) -> int:
    pid_file = _pid_path(state_dir)
    pid = _read_pid(pid_file)
    if not pid:
        print("No running UI found (missing pid file).")
        return 1
    if not _is_running(pid):
        pid_file.unlink(missing_ok=True)
        print("UI not running (stale pid file removed).")
        return 0

    os.kill(pid, signal.SIGINT)
    if _wait_for_exit(pid, timeout_s=5.0):
        pid_file.unlink(missing_ok=True)
        print("UI stopped.")
        return 0

    os.kill(pid, signal.SIGTERM)
    if _wait_for_exit(pid, timeout_s=5.0):
        pid_file.unlink(missing_ok=True)
        print("UI stopped.")
        return 0

    print("Failed to stop UI. Try `kill` manually.")
    return 2


def _status(state_dir: Path) -> int:
    pid_file = _pid_path(state_dir)
    pid = _read_pid(pid_file)
    if not pid:
        print("UI not running.")
        return 1
    if _is_running(pid):
        print(f"UI running (pid {pid}).")
        print(f"Log: {_log_path(state_dir)}")
        return 0
    pid_file.unlink(missing_ok=True)
    print("UI not running (stale pid file removed).")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the stEVE NiceGUI UI process.")
    parser.add_argument(
        "command",
        choices=["start", "stop", "status"],
        help="Action to perform.",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for pid/log files (defaults to XDG_STATE_HOME).",
    )
    args = parser.parse_args()

    state_dir = Path(args.state_dir).expanduser() if args.state_dir else _state_root()
    if args.command == "start":
        raise SystemExit(_start(state_dir))
    if args.command == "stop":
        raise SystemExit(_stop(state_dir))
    raise SystemExit(_status(state_dir))


if __name__ == "__main__":
    main()
