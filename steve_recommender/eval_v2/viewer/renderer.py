"""Shared PyVista scene renderer for eval_v2 force-trace replay."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pyvista as pv

from .trace_data import TraceData, TraceFrame


BOTTOM_RIGHT_SCALAR_BAR_POSITION_X = 0.72
BOTTOM_RIGHT_SCALAR_BAR_POSITION_Y = 0.04
BOTTOM_RIGHT_SCALAR_BAR_WIDTH = 0.25
BOTTOM_RIGHT_SCALAR_BAR_HEIGHT = 0.04
MAX_FORCE_READOUT_PADDING_PX = 10
MAX_FORCE_READOUT_FONT_SIZE = 10
NEUTRAL_VESSEL_COLOR_RGB = "#8B949E"
WIRE_COLOR_RGB = "#F2AC32"
REPLAY_BACKGROUND_RGB = "#2A2A2A"
SCALAR_BAR_TITLE = "Force (N)"
STEP_TEXT_POSITION = "upper_left"
MAX_FORCE_READOUT_POSITION = "upper_right"


class TraceRenderer:
    """Render one `TraceData` object into a standalone or injected PyVista plotter.

    Parameters use SI Newtons for force display and millimeters for geometry,
    matching the persisted Phase E trace schema. When
    `max_display_force_N` is omitted the renderer uses the trace loader's
    percentile-calibrated recommendation so typical contact variation stays
    visible even when a trace contains occasional high-force outliers.
    """

    def __init__(
        self,
        trace: TraceData,
        max_display_force_N: Optional[float] = None,
        plotter: Optional[pv.Plotter] = None,
        off_screen: bool = False,
    ) -> None:
        self._trace = trace
        self._max_display_force_N = float(
            trace.recommended_max_display_force_N
            if max_display_force_N is None
            else max_display_force_N
        )
        if (
            plotter is None
            and not os.environ.get("DISPLAY")
            and hasattr(pv, "start_xvfb")
        ):
            try:
                pv.start_xvfb()
            except Exception:
                pass
        self._plotter = plotter or pv.Plotter(
            off_screen=off_screen,
            window_size=(640, 480),
        )
        self._owns_plotter = plotter is None
        self._mesh = pv.read(str(self._trace.vessel_mesh_path))
        self._wire_actor = None
        self._contact_actor = None
        self._text_actor = None
        self._max_force_readout_actor = None
        self._max_force_readout_text = "Max force: 0.00 N"
        self.current_frame: Optional[TraceFrame] = None
        self._scalar_bar_args = {
            "title": SCALAR_BAR_TITLE,
            "position_x": BOTTOM_RIGHT_SCALAR_BAR_POSITION_X,
            "position_y": BOTTOM_RIGHT_SCALAR_BAR_POSITION_Y,
            "width": BOTTOM_RIGHT_SCALAR_BAR_WIDTH,
            "height": BOTTOM_RIGHT_SCALAR_BAR_HEIGHT,
        }
        self.current_step = 0

        self._plotter.set_background(REPLAY_BACKGROUND_RGB)
        self._plotter.add_mesh(
            self._mesh,
            color=NEUTRAL_VESSEL_COLOR_RGB,
            opacity=0.35,
            smooth_shading=True,
        )
        self._plotter.reset_camera()
        self._text_actor = self._plotter.add_text(
            "",
            position=STEP_TEXT_POSITION,
            font_size=10,
            color="white",
        )
        self._max_force_readout_actor = self._plotter.add_text(
            "",
            position=MAX_FORCE_READOUT_POSITION,
            font_size=MAX_FORCE_READOUT_FONT_SIZE,
            color="white",
        )

    def render_frame(self, step: int) -> None:
        """Render one frame for simulation step `step`."""

        clamped_step = self._clamp_step(step)
        frame = self._trace.frame(clamped_step)
        self.current_frame = frame
        self.current_step = int(frame.step_index)
        self._update_wire(frame.wire_positions_mm)
        self._update_triangle_forces(
            frame.triangle_force_indices,
            frame.triangle_force_magnitudes_N,
        )
        self._update_step_text(frame.step_index)
        self._update_max_force_readout(frame.triangle_force_magnitudes_N)
        self._plotter.render()

    def get_plotter(self) -> pv.Plotter:
        """Return the underlying PyVista plotter."""

        return self._plotter

    def close(self) -> None:
        """Release renderer resources and close owned plotter windows."""

        if self._owns_plotter:
            self._plotter.close()

    def _update_wire(self, wire_positions_mm: np.ndarray) -> None:
        polyline = pv.lines_from_points(
            np.asarray(wire_positions_mm, dtype=np.float32),
            close=False,
        )
        if self._wire_actor is not None:
            self._plotter.remove_actor(self._wire_actor)
        self._wire_actor = self._plotter.add_mesh(
            polyline,
            color=WIRE_COLOR_RGB,
            line_width=4,
        )

    def _update_triangle_forces(
        self,
        triangle_indices: np.ndarray,
        triangle_magnitudes_N: np.ndarray,
    ) -> None:
        if self._contact_actor is not None:
            self._plotter.remove_actor(self._contact_actor)
            self._contact_actor = None

        if triangle_indices.size == 0:
            return

        highlighted = self._mesh.extract_cells(
            np.asarray(triangle_indices, dtype=np.int64)
        ).copy()
        highlighted.cell_data["triangle_force_magnitudes_N"] = np.asarray(
            triangle_magnitudes_N,
            dtype=np.float32,
        )
        self._contact_actor = self._plotter.add_mesh(
            highlighted,
            scalars="triangle_force_magnitudes_N",
            cmap="plasma",
            clim=(0.0, self._max_display_force_N),
            show_scalar_bar=True,
            scalar_bar_args=self._scalar_bar_args,
        )

    def _update_step_text(self, step_index: int) -> None:
        if self._text_actor is not None and hasattr(self._text_actor, "SetText"):
            self._text_actor.SetText(2, f"Step {int(step_index)}")
        else:
            if self._text_actor is not None:
                self._plotter.remove_actor(self._text_actor)
            self._text_actor = self._plotter.add_text(
                f"Step {int(step_index)}",
                position=STEP_TEXT_POSITION,
                font_size=10,
                color="white",
            )

    def _update_max_force_readout(self, triangle_magnitudes_N: np.ndarray) -> None:
        max_force_N = (
            float(np.max(triangle_magnitudes_N))
            if triangle_magnitudes_N.size > 0
            else 0.0
        )
        self._max_force_readout_text = f"Max force: {max_force_N:.2f} N"
        if self._max_force_readout_actor is not None and hasattr(
            self._max_force_readout_actor, "SetText"
        ):
            self._max_force_readout_actor.SetText(3, self._max_force_readout_text)
        else:
            if self._max_force_readout_actor is not None:
                self._plotter.remove_actor(self._max_force_readout_actor)
            self._max_force_readout_actor = self._plotter.add_text(
                self._max_force_readout_text,
                position=MAX_FORCE_READOUT_POSITION,
                font_size=MAX_FORCE_READOUT_FONT_SIZE,
                color="white",
            )

    def _clamp_step(self, step: int) -> int:
        return max(0, min(int(step), self._trace.n_steps - 1))
