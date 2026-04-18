from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .scene_factory import apply_camera_pose


class SofaSceneViewer:
    """Optional SOFA scene viewer for the force playground loop."""

    def __init__(
        self,
        *,
        enabled: bool,
        intervention: Any,
        reset_seed: Optional[int] = None,
        camera_position: Optional[np.ndarray] = None,
        camera_look_at: Optional[np.ndarray] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.should_quit = False
        self._intervention = intervention
        self._reset_seed = reset_seed
        self._camera_position = (
            np.asarray(camera_position, dtype=np.float32).reshape((3,))
            if camera_position is not None
            else None
        )
        self._camera_look_at = (
            np.asarray(camera_look_at, dtype=np.float32).reshape((3,))
            if camera_look_at is not None
            else None
        )
        self._viewer: Optional[Any] = None
        self._initialized = False
        self.last_error = ""

        if not self.enabled:
            return

        from steve_recommender.visualisation.interactive_sofapygame import (
            InteractiveSofaPygame,
        )

        self._viewer = InteractiveSofaPygame(intervention=intervention)
        self._init_viewer()

    def _simulation_has_camera(self) -> bool:
        try:
            sim = self._intervention.simulation
            camera = getattr(sim, "camera", None)
            return camera is not None
        except Exception:
            return False

    def _apply_camera_pose(self) -> None:
        if self._camera_position is None or self._camera_look_at is None:
            return
        try:
            apply_camera_pose(
                self._intervention.simulation,
                position=self._camera_position,
                look_at=self._camera_look_at,
            )
        except Exception:
            pass

    def _reset_intervention_for_visuals(self) -> None:
        # stEVE's SofaBeamAdapter.reset() does not rebuild the graph when only
        # init_visual_nodes changes. For a reliable camera setup we force a full
        # rebuild once, then run intervention.reset().
        try:
            sim = self._intervention.simulation
            unload = getattr(sim, "_unload_simulation", None)
            if callable(unload):
                unload()
            if hasattr(sim, "root"):
                sim.root = None
            if hasattr(sim, "camera"):
                sim.camera = None
        except Exception:
            pass

        seed = self._reset_seed
        if seed is None:
            self._intervention.reset()
        else:
            self._intervention.reset(seed=int(seed))
        self._apply_camera_pose()

    def _init_viewer(self) -> None:
        if self._viewer is None or self._initialized:
            return

        # If the scene was already reset without visual nodes, camera can be missing.
        # Force one reset after viewer creation to allocate visual objects/camera.
        if not self._simulation_has_camera():
            try:
                self._reset_intervention_for_visuals()
            except Exception:
                pass

        try:
            # Important: SofaPygame needs reset() before first render()
            # to initialize pygame display and OpenGL context.
            self._viewer.reset(episode_nr=0)
            self._apply_camera_pose()
            self._initialized = True
        except Exception as exc:
            # One recovery attempt after an explicit intervention reset.
            self.last_error = str(exc)
            try:
                self._reset_intervention_for_visuals()
                self._viewer.reset(episode_nr=0)
                self._apply_camera_pose()
                self._initialized = True
                self.last_error = ""
                return
            except Exception as exc_retry:
                # Keep playground alive even if scene window cannot be initialized.
                self.should_quit = True
                self.last_error = f"{exc} | retry={exc_retry}"

    def poll_events(self) -> None:
        if not self.enabled or self._viewer is None:
            return
        if not self._initialized:
            self._init_viewer()
            if not self._initialized:
                return
        try:
            poll_fn = getattr(self._viewer, "poll_events", None)
            if callable(poll_fn):
                poll_fn()
                key_fn = getattr(self._viewer, "apply_keyboard_controls", None)
                if callable(key_fn):
                    key_fn()
            else:
                # Fallback path for older viewer adapters.
                self._viewer.render()
        except KeyboardInterrupt:
            self.should_quit = True
        except Exception as exc:
            self.should_quit = True
            self.last_error = str(exc)

    def render_frame(self) -> None:
        if not self.enabled or self._viewer is None:
            return
        if not self._initialized:
            self._init_viewer()
            if not self._initialized:
                return
        try:
            self._viewer.render()
        except KeyboardInterrupt:
            self.should_quit = True
        except Exception as exc:
            self.should_quit = True
            self.last_error = str(exc)

    def render(self) -> None:
        # Backward-compatible alias.
        self.render_frame()

    def camera_state(self) -> dict[str, list[float] | None]:
        try:
            cam = self._intervention.simulation.camera
        except Exception:
            return {"position": None, "look_at": None}
        if cam is None:
            return {"position": None, "look_at": None}
        position = getattr(cam, "position", None)
        look_at = getattr(cam, "lookAt", None)
        if hasattr(position, "value"):
            position = position.value
        if hasattr(look_at, "value"):
            look_at = look_at.value
        try:
            position = [float(x) for x in np.asarray(position).reshape((3,))]
        except Exception:
            position = None
        try:
            look_at = [float(x) for x in np.asarray(look_at).reshape((3,))]
        except Exception:
            look_at = None
        return {"position": position, "look_at": look_at}

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
