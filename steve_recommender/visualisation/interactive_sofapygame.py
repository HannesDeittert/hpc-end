"""Interactive SofaPygame viewer.

This wraps the standard eve.visualisation.SofaPygame visualiser and adds
simple keyboard controls for rotating and zooming the camera:

- Arrow left/right  : rotate LAO/RAO
- Arrow up/down     : rotate CRA/CAU
- W / S             : zoom in / out
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from steve_recommender.steve_adapter import eve

@dataclass
class _KeyConfig:
    # Camera rotation speed (degrees-equivalent per second; higher = faster)
    rotate_speed: float = 4.0
    # Zoom speed (translation units per second; higher = faster)
    zoom_speed: float = 160.0


class InteractiveSofaPygame(eve.visualisation.SofaPygame):
    """SofaPygame with basic keyboard-driven camera control."""

    def __init__(
        self,
        intervention: eve.intervention.Intervention,
        interim_target: Optional[eve.interimtarget.InterimTarget] = None,
        display_size: Tuple[float, float] = (600, 860),
        color: Tuple[float, float, float, float] = (0, 0, 0, 0),
        key_config: Optional[_KeyConfig] = None,
    ) -> None:
        super().__init__(
            intervention=intervention,
            interim_target=interim_target,
            display_size=display_size,
            color=color,
        )
        self._keys = key_config or _KeyConfig()
        self._pygame = None

    def _ensure_pygame(self):
        if self._pygame is None:
            self._pygame = importlib.import_module("pygame")

    def _handle_keyboard(self) -> None:
        """Apply keyboard-driven camera updates before rendering."""
        self._ensure_pygame()
        pg = self._pygame

        # Drain events so the window stays responsive (close button etc.).
        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise KeyboardInterrupt()

        keys = pg.key.get_pressed()
        lao_rao = 0.0
        cra_cau = 0.0
        zoom = 0.0

        if keys[pg.K_LEFT]:
            lao_rao -= self._keys.rotate_speed
        if keys[pg.K_RIGHT]:
            lao_rao += self._keys.rotate_speed
        if keys[pg.K_UP]:
            cra_cau += self._keys.rotate_speed
        if keys[pg.K_DOWN]:
            cra_cau -= self._keys.rotate_speed

        if keys[pg.K_w]:
            zoom += self._keys.zoom_speed
        if keys[pg.K_s]:
            zoom -= self._keys.zoom_speed

        if lao_rao or cra_cau:
            self.rotate(lao_rao, cra_cau)
        if zoom:
            self.zoom(zoom)

    def render(self) -> np.ndarray:
        # Apply keyboard controls, then delegate to the base implementation.
        try:
            self._handle_keyboard()
        except KeyboardInterrupt:
            # Propagate upwards so the caller can stop the episode cleanly.
            raise
        return super().render()
