from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class LivePlotter:
    def __init__(
        self,
        *,
        enabled: bool,
        interactive: bool,
        snapshot_dir: Optional[Path] = None,
        on_adjust_key: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.interactive = bool(interactive)
        self.snapshot_dir = snapshot_dir
        self.on_adjust_key = on_adjust_key

        self.should_quit = False
        self.run_continuous = not bool(interactive)
        self._pending_steps = 0
        self._status_note = ""
        self._snapshot_requested = False
        self._snapshot_idx = 0

        self._plt = None
        self._fig = None
        self._lines: Dict[str, Any] = {}
        self._x: List[int] = []
        self._series: Dict[str, List[float]] = {
            "norm_sum_vector": [],
            "sum_norm": [],
            "peak_triangle_force": [],
            "sum_abs_fn": [],
            "sum_abs_ft": [],
            "lambda_abs_sum": [],
            "lambda_dt_abs_sum": [],
            "lambda_active_rows_count": [],
        }

        if self.enabled:
            self._init_plot()

    def _import_pyplot_with_safe_backend(self):
        """Import matplotlib.pyplot with a non-Qt default to avoid xcb aborts.

        We intentionally avoid Qt backends by default because some conda/OpenCV
        setups export incompatible Qt plugin paths, which can hard-abort the
        process on import (not a recoverable Python exception).
        """

        import matplotlib

        # Allow explicit override for debugging.
        forced = str(os.environ.get("STEVE_FORCE_PLAYGROUND_MPL_BACKEND", "")).strip()
        if forced:
            candidates = [forced]
        else:
            # Prefer Tk for interactive usage; always keep Agg as safe fallback.
            candidates = ["TkAgg", "Agg"] if self.interactive else ["Agg", "TkAgg"]

        last_err: Optional[Exception] = None
        for backend in candidates:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                # Agg cannot provide keyboard interaction; degrade gracefully.
                if backend.lower() == "agg" and self.interactive:
                    self.interactive = False
                    self.run_continuous = True
                    self._status_note = "Agg backend active: interactive hotkeys disabled"
                return plt
            except Exception as exc:
                last_err = exc
                continue

        msg = "Failed to initialize matplotlib backend"
        if last_err is not None:
            msg += f": {last_err}"
        raise RuntimeError(msg)

    def _init_plot(self) -> None:
        plt = self._import_pyplot_with_safe_backend()
        self._plt = plt
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()

        (l_norm_sum_vector,) = ax1.plot([], [], label="norm(sum(v_i))")
        (l_sum_norm,) = ax1.plot([], [], label="sum(norm(v_i))")
        (l_peak,) = ax1.plot([], [], label="peak_triangle_force")
        ax1.set_title("Force Aggregates")
        ax1.set_xlabel("Step")
        ax1.legend(loc="upper left")

        (l_fn,) = ax2.plot([], [], label="|F_n| (sum)")
        (l_ft,) = ax2.plot([], [], label="|F_t| (sum)")
        ax2.set_title("Normal vs Tangential")
        ax2.set_xlabel("Step")
        ax2.legend(loc="upper left")

        (l_lambda,) = ax3.plot([], [], label="sum(|lambda|)")
        (l_lambda_dt,) = ax3.plot([], [], label="sum(|lambda/dt|)")
        ax3.set_title("Constraint Magnitudes")
        ax3.set_xlabel("Step")
        ax3.legend(loc="upper left")

        (l_active_rows,) = ax4.plot([], [], label="active lambda rows")
        ax4.set_title("Active Rows")
        ax4.set_xlabel("Step")
        ax4.legend(loc="upper left")

        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig = fig
        self._lines = {
            "norm_sum_vector": l_norm_sum_vector,
            "sum_norm": l_sum_norm,
            "peak_triangle_force": l_peak,
            "sum_abs_fn": l_fn,
            "sum_abs_ft": l_ft,
            "lambda_abs_sum": l_lambda,
            "lambda_dt_abs_sum": l_lambda_dt,
            "lambda_active_rows_count": l_active_rows,
        }
        self._set_title()
        plt.tight_layout()
        plt.pause(0.001)

    def _set_title(self) -> None:
        if not self.enabled or self._fig is None:
            return
        mode = "RUN" if self.run_continuous else "PAUSE"
        text = "force_playground"
        text += " | hotkeys: space/n=step c=run/pause q=quit p=snapshot up/down=insert +/-=force"
        text += f" | mode={mode}"
        if self._status_note:
            text += f" | {self._status_note}"
        self._fig.suptitle(text, fontsize=10)

    def _on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower().strip()
        if not key:
            return
        if key in {"q", "escape"}:
            self.should_quit = True
            self._status_note = "quit requested"
        elif key in {" ", "space", "n", "enter", "return"}:
            self._pending_steps += 1
            self._status_note = f"step queued ({self._pending_steps})"
        elif key == "c":
            self.run_continuous = not self.run_continuous
            self._status_note = "continuous on" if self.run_continuous else "continuous off"
        elif key == "p":
            self._snapshot_requested = True
            self._status_note = "snapshot requested"
        elif key in {"up", "down", "+", "=", "-"} and self.on_adjust_key is not None:
            msg = self.on_adjust_key(key)
            if msg:
                self._status_note = msg
        self._set_title()

    def wait_until_step_allowed(
        self,
        *,
        idle_callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        if not self.enabled:
            return True
        if not self.interactive:
            return True
        while not self.should_quit:
            if self.run_continuous:
                return True
            if self._pending_steps > 0:
                self._pending_steps -= 1
                return True
            if idle_callback is not None:
                try:
                    idle_callback()
                except KeyboardInterrupt:
                    self.should_quit = True
                    self._status_note = "quit requested"
                except Exception:
                    # Keep plotting loop alive if idle callback fails once.
                    pass
            self._set_title()
            self._plt.pause(0.05)
        return False

    def update(self, step_record: Dict[str, float]) -> None:
        if not self.enabled:
            return

        step_idx = int(step_record.get("step", 0))
        self._x.append(step_idx)
        for key in self._series.keys():
            self._series[key].append(float(step_record.get(key, 0.0)))

        for key, line in self._lines.items():
            line.set_data(self._x, self._series[key])
            ax = line.axes
            ax.relim()
            ax.autoscale_view()

        self._set_title()
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

        if self._snapshot_requested and self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshot_dir / f"plot_{self._snapshot_idx:05d}.png"
            self._fig.savefig(path, dpi=140)
            self._snapshot_idx += 1
            self._snapshot_requested = False
            self._status_note = f"snapshot saved: {path.name}"
            self._set_title()

    def close(self) -> None:
        if self.enabled and self._plt is not None and self._fig is not None:
            self._plt.ioff()
