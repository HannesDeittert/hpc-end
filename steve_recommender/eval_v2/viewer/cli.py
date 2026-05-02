"""Standalone CLI host for the eval_v2 trace replay viewer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence, TextIO

from PyQt5.QtWidgets import QApplication, QMainWindow

from .qt_replay_widget import QtTraceReplayWidget


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone replay-viewer CLI parser."""

    parser = argparse.ArgumentParser(
        description="Open an eval_v2 force-trace replay viewer."
    )
    parser.add_argument(
        "path", help="Trace .h5 path or job directory containing traces/"
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-force-n", type=float, default=None)
    parser.add_argument("--off-screen", action="store_true", help=argparse.SUPPRESS)
    return parser


def resolve_trace_path(path_arg: str) -> Path:
    """Resolve a trace file path from either a direct file or a job directory."""

    path = Path(path_arg)
    if not path.exists():
        raise FileNotFoundError(f"Trace path does not exist: {path}")
    if path.is_dir():
        traces_dir = path / "traces"
        if not traces_dir.exists():
            raise FileNotFoundError(f"Job directory has no traces/ folder: {path}")
        traces = sorted(traces_dir.glob("*.h5"))
        if not traces:
            raise FileNotFoundError(
                f"Job directory has no trace .h5 files: {traces_dir}"
            )
        return traces[0]
    if path.suffix.lower() != ".h5":
        raise ValueError(f"Trace path must be a .h5 file, got: {path}")
    return path


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: TextIO,
    stderr: Optional[TextIO] = None,
) -> int:
    """Run the standalone replay-viewer CLI."""

    err = stderr or stdout
    try:
        args = build_parser().parse_args(argv)
        trace_path = resolve_trace_path(args.path)
        app = QApplication.instance()
        if app is None:
            app_args = ["eval_v2_viewer"]
            if bool(args.off_screen):
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                app_args.extend(["-platform", "offscreen"])
            app = QApplication(app_args)
        widget = QtTraceReplayWidget()
        try:
            widget.open_trace(
                trace_path,
                start_step=int(args.start_step),
                max_force_n=args.max_force_n,
            )
            stdout.write(f"Opened trace viewer for {trace_path}\n")
            if bool(args.off_screen):
                widget.close_trace()
                widget.close()
                return 0

            window = QMainWindow()
            window.setWindowTitle(f"eval_v2 Trace Viewer - {trace_path.name}")
            window.setCentralWidget(widget)
            window.resize(1200, 800)
            window.show()
            return int(app.exec_())
        finally:
            if bool(args.off_screen):
                widget.close()
        return 0
    except Exception as exc:
        err.write(f"{exc}\n")
        return 2


__all__ = ["build_parser", "main", "resolve_trace_path"]
