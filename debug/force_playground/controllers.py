from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import ForcePlaygroundConfig


@dataclass
class StepCommand:
    action: np.ndarray
    commanded_force_vector_n: np.ndarray
    commanded_force_scalar_n: float
    controller_status: str


class ForceApplicator:
    """Best-effort external-force injector for open-loop rigid-probe tests."""

    def __init__(self, simulation: Any, *, node_index: int = 0) -> None:
        self._simulation = simulation
        self._node_index = int(node_index)
        self._target_field = ""
        self._last_force_scene = np.zeros((3,), dtype=np.float32)
        self._last_force_local = np.zeros((3,), dtype=np.float32)
        # Debug switch:
        # Some SOFA objects expose node frames where externalForce is interpreted
        # in world coordinates, not local coordinates. Keep local-frame conversion
        # default-on for backward compatibility, but allow disabling to validate
        # force-direction assumptions without code edits.
        self._use_local_frame = str(
            os.environ.get("STEVE_FORCE_APPLICATOR_LOCAL_FRAME", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _quat_xyzw_to_rotation(q_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_xyzw, dtype=np.float32).reshape((4,))
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 1e-12:
            return np.eye(3, dtype=np.float32)
        x, y, z, w = (q / n).tolist()
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.asarray(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _read_data(obj: Any, attr: str) -> Any:
        try:
            value = getattr(obj, attr)
        except Exception:
            return None
        if hasattr(value, "value"):
            try:
                return value.value
            except Exception:
                return None
        return value

    @staticmethod
    def _set_data(obj: Any, attr: str, value: Any) -> bool:
        try:
            data = getattr(obj, attr)
            if hasattr(data, "value"):
                data.value = value
            else:
                setattr(obj, attr, value)
            return True
        except Exception:
            pass
        try:
            data = obj.findData(attr)
            data.value = value
            return True
        except Exception:
            return False

    def _wire_dofs(self) -> Optional[Any]:
        sim = self._simulation
        try:
            return sim._instruments_combined.DOFs  # noqa: SLF001
        except Exception:
            return None

    def apply_force_scene(self, force_vec_scene: np.ndarray) -> str:
        dofs = self._wire_dofs()
        if dofs is None:
            return "wire_dofs_missing"

        pos = self._read_data(dofs, "position")
        pos_arr = np.asarray(pos) if pos is not None else np.zeros((0, 3), dtype=np.float32)
        if pos_arr.ndim == 1:
            if pos_arr.size % 3 == 0:
                pos_arr = pos_arr.reshape((-1, 3))
            else:
                pos_arr = pos_arr.reshape((1, -1))
        n_nodes = int(pos_arr.shape[0]) if pos_arr.size else 0
        if n_nodes <= 0:
            return "wire_positions_unavailable"

        if self._node_index < 0:
            idx = n_nodes + int(self._node_index)
        else:
            idx = int(self._node_index)
        idx = int(min(max(idx, 0), n_nodes - 1))

        force_world = np.asarray(force_vec_scene, dtype=np.float32).reshape((3,))
        force_local = force_world.copy()
        # Beam DOFs are represented in node-local frames.
        # Convert requested world force to local coordinates using node quaternion.
        if self._use_local_frame and pos_arr.ndim == 2 and pos_arr.shape[1] >= 7:
            q_xyzw = np.asarray(pos_arr[idx, 3:7], dtype=np.float32).reshape((4,))
            rot = self._quat_xyzw_to_rotation(q_xyzw)
            force_local = (rot.T @ force_world).astype(np.float32)

        # Preserve layout of existing force field if possible.
        template = self._read_data(dofs, "externalForce")
        field_name = "externalForce"
        if template is None:
            template = self._read_data(dofs, "force")
            field_name = "force"
        if template is None:
            # Fallback to Nx3 array.
            arr = np.zeros((n_nodes, 3), dtype=np.float32)
            arr[idx, :3] = force_local
            if self._set_data(dofs, "externalForce", arr):
                self._target_field = "externalForce"
                self._last_force_scene = force_world
                self._last_force_local = force_local
                return "applied:externalForce"
            if self._set_data(dofs, "force", arr):
                self._target_field = "force"
                self._last_force_scene = force_world
                self._last_force_local = force_local
                return "applied:force"
            return "force_field_missing"

        arr = np.asarray(template)
        if arr.ndim == 1 and arr.size % n_nodes == 0:
            width = int(arr.size // n_nodes)
            arr2 = np.zeros((n_nodes, width), dtype=np.float32)
            arr2[idx, 0:3] = force_local
            payload = arr2.reshape((-1,))
        else:
            if arr.ndim == 1:
                arr = arr.reshape((1, -1))
            width = int(arr.shape[1]) if arr.ndim >= 2 else 3
            arr2 = np.zeros((n_nodes, max(width, 3)), dtype=np.float32)
            arr2[idx, 0:3] = force_local
            payload = arr2

        if not self._set_data(dofs, field_name, payload):
            return f"apply_failed:{field_name}"

        self._target_field = field_name
        self._last_force_scene = force_world
        self._last_force_local = force_local
        return f"applied:{field_name}"

    def clear(self) -> None:
        dofs = self._wire_dofs()
        if dofs is None:
            return
        field_name = self._target_field or "externalForce"
        template = self._read_data(dofs, field_name)
        if template is None:
            return
        arr = np.asarray(template)
        zeros = np.zeros_like(arr, dtype=np.float32)
        self._set_data(dofs, field_name, zeros)
        self._last_force_scene = np.zeros((3,), dtype=np.float32)
        self._last_force_local = np.zeros((3,), dtype=np.float32)


class BaseController:
    def __init__(self, cfg: ForcePlaygroundConfig, wall_reference_normal: np.ndarray) -> None:
        self._cfg = cfg
        self._wall_reference_normal = np.asarray(wall_reference_normal, dtype=np.float32).reshape((3,))
        self.insert_action = float(cfg.control.insert_action)
        self.rotate_action = float(cfg.control.rotate_action)
        self.target_force_n = float(cfg.control.open_loop_force_n)

    def on_key(self, key: str) -> str:
        if key == "up":
            self.insert_action += float(self._cfg.control.action_step_delta)
            return f"insert_action={self.insert_action:.4f}"
        if key == "down":
            self.insert_action -= float(self._cfg.control.action_step_delta)
            return f"insert_action={self.insert_action:.4f}"
        if key in {"+", "="}:
            self.target_force_n += float(self._cfg.control.force_step_delta_n)
            return f"target_force_n={self.target_force_n:.4f}"
        if key == "-":
            self.target_force_n = max(0.0, self.target_force_n - float(self._cfg.control.force_step_delta_n))
            return f"target_force_n={self.target_force_n:.4f}"
        return ""

    def command(self, step_index: int) -> StepCommand:
        raise NotImplementedError


class DisplacementController(BaseController):
    def command(self, step_index: int) -> StepCommand:
        _ = step_index
        action = np.asarray([[self.insert_action, self.rotate_action]], dtype=np.float32)
        return StepCommand(
            action=action,
            commanded_force_vector_n=np.zeros((3,), dtype=np.float32),
            commanded_force_scalar_n=0.0,
            controller_status="displacement",
        )


class OpenLoopForceController(BaseController):
    def __init__(self, cfg: ForcePlaygroundConfig, wall_reference_normal: np.ndarray) -> None:
        super().__init__(cfg, wall_reference_normal)
        # Keep open-loop displacement channel independent from the generic
        # displacement controller default.
        self.insert_action = float(cfg.control.open_loop_insert_action)

    def command(self, step_index: int) -> StepCommand:
        _ = step_index
        action = np.asarray(
            [[float(self.insert_action), self.rotate_action]],
            dtype=np.float32,
        )
        n = np.asarray(self._wall_reference_normal, dtype=np.float32).reshape((3,))
        # Push probe into wall: applied probe force opposite to wall outward normal.
        force_vec_n = -float(self.target_force_n) * n
        return StepCommand(
            action=action,
            commanded_force_vector_n=force_vec_n.astype(np.float32),
            commanded_force_scalar_n=float(self.target_force_n),
            controller_status="open_loop_force",
        )


def build_controller(cfg: ForcePlaygroundConfig, wall_reference_normal: np.ndarray) -> BaseController:
    if cfg.mode == "open_loop_force":
        if cfg.probe != "rigid_probe":
            raise ValueError("open_loop_force is supported only for probe='rigid_probe' in v1")
        return OpenLoopForceController(cfg, wall_reference_normal)
    return DisplacementController(cfg, wall_reference_normal)
