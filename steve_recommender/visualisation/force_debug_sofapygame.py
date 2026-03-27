"""Sofa/Pygame visualiser with live wall-force debug telemetry.

This viewer is intentionally lightweight:
- camera controls from InteractiveSofaPygame stay available
- a compact force/contact summary is shown in the window title
- segment-force peaks are printed to stdout for step-wise debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interactive_sofapygame import InteractiveSofaPygame


@dataclass
class _ForceDebugConfig:
    top_k_segments: int = 5
    print_every_steps: int = 10


class ForceDebugSofaPygame(InteractiveSofaPygame):
    """Interactive SofaPygame with per-step force debug output."""

    def __init__(
        self,
        intervention,
        interim_target=None,
        *,
        force_info=None,
        display_size: Tuple[float, float] = (600, 860),
        color: Tuple[float, float, float, float] = (0, 0, 0, 0),
        top_k_segments: int = 5,
        print_every_steps: int = 10,
    ) -> None:
        super().__init__(
            intervention=intervention,
            interim_target=interim_target,
            display_size=display_size,
            color=color,
        )
        self._force_info = force_info
        self._cfg = _ForceDebugConfig(
            top_k_segments=max(1, int(top_k_segments)),
            print_every_steps=max(1, int(print_every_steps)),
        )
        self._step_idx = 0
        self._last_logged_signature: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def _float_or_nan(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _segment_top_k(
        segment_vectors: Any,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        try:
            seg = np.asarray(segment_vectors, dtype=np.float32).reshape((-1, 3))
        except Exception:
            return []
        if seg.size == 0:
            return []
        norms = np.linalg.norm(seg, axis=1)
        if norms.size == 0:
            return []
        order = np.argsort(norms)[::-1]
        out: List[Tuple[int, float]] = []
        for idx in order[:top_k]:
            val = float(norms[idx])
            if not np.isfinite(val):
                continue
            out.append((int(idx), val))
        return out

    @staticmethod
    def _active_segment_top_k(
        active_ids: Any,
        active_vectors: Any,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        try:
            ids = np.asarray(active_ids, dtype=np.int64).reshape((-1,))
            vecs = np.asarray(active_vectors, dtype=np.float32).reshape((-1, 3))
        except Exception:
            return []
        m = min(ids.shape[0], vecs.shape[0])
        if m <= 0:
            return []
        ids = ids[:m]
        norms = np.linalg.norm(vecs[:m], axis=1)
        order = np.argsort(norms)[::-1]
        out: List[Tuple[int, float]] = []
        for idx in order[:top_k]:
            val = float(norms[idx])
            if not np.isfinite(val):
                continue
            out.append((int(ids[idx]), val))
        return out

    def _read_force_debug(self) -> Dict[str, Any]:
        if self._force_info is None:
            return {}
        try:
            info = self._force_info.info
            if isinstance(info, dict):
                return info
        except Exception:
            pass
        return {}

    @staticmethod
    def _format_force(x: float) -> str:
        if not np.isfinite(x):
            return "nan"
        return f"{x:.3g}"

    @staticmethod
    def _format_top_segments(values: List[Tuple[int, float]]) -> str:
        if not values:
            return "-"
        return ", ".join(f"s{i}:{v:.3g}" for i, v in values)

    def _update_debug_outputs(self) -> None:
        self._ensure_pygame()
        pg = self._pygame
        info = self._read_force_debug()

        available = bool(info.get("wall_force_available", False))
        source = str(info.get("wall_force_source", "")).strip() or "unknown"
        error = str(info.get("wall_force_error", "")).strip()
        status = str(info.get("wall_force_status", "")).strip()
        if not error and status.startswith("err:"):
            error = status
        contact_count = self._safe_int(info.get("wall_contact_count", 0), default=0)
        contact_detected = bool(info.get("wall_contact_detected", False))
        force_channel = str(info.get("wall_force_channel", "")).strip() or "none"
        wire_force_src = str(info.get("wall_wire_force_vectors_source", "")).strip() or "-"
        coll_force_src = (
            str(info.get("wall_collision_force_vectors_source", "")).strip() or "-"
        )
        force_norm_sum = self._float_or_nan(info.get("wall_force_norm_sum", float("nan")))
        total_norm = self._float_or_nan(info.get("wall_total_force_norm", float("nan")))
        wire_norm = self._float_or_nan(info.get("wall_wire_force_norm", float("nan")))
        collision_norm = self._float_or_nan(
            info.get("wall_collision_force_norm", float("nan"))
        )
        lcp_max_abs = self._float_or_nan(info.get("wall_lcp_max_abs", float("nan")))
        top_segments = self._active_segment_top_k(
            info.get("wall_active_segment_ids", np.zeros((0,), dtype=np.int32)),
            info.get(
                "wall_active_segment_force_vectors",
                np.zeros((0, 3), dtype=np.float32),
            ),
            top_k=self._cfg.top_k_segments,
        )
        if not top_segments:
            top_segments = self._segment_top_k(
                info.get("wall_segment_force_vectors", np.zeros((0, 3), dtype=np.float32)),
                top_k=self._cfg.top_k_segments,
            )
        top_segments_str = self._format_top_segments(top_segments)

        caption = (
            "force_debug | "
            f"step={self._step_idx} "
            f"contact={contact_count} "
            f"det={int(contact_detected)} "
            f"avail={int(available)} "
            f"Ftot={self._format_force(total_norm)} "
            f"Fwire={self._format_force(wire_norm)} "
            f"Fcoll={self._format_force(collision_norm)} "
            f"Fsum={self._format_force(force_norm_sum)} "
            f"LCPmax={self._format_force(lcp_max_abs)} "
            f"source={source} "
            f"chan={force_channel} "
            f"wireSrc={wire_force_src} "
            f"collSrc={coll_force_src} "
            f"top={top_segments_str}"
        )
        if error:
            caption += f" err={error[:60]}"
        pg.display.set_caption(caption)

        signature = (
            contact_count,
            int(contact_detected),
            int(available),
            round(total_norm, 8) if np.isfinite(total_norm) else "nan",
            round(wire_norm, 8) if np.isfinite(wire_norm) else "nan",
            round(collision_norm, 8) if np.isfinite(collision_norm) else "nan",
            source,
            force_channel,
            tuple((i, round(v, 8)) for i, v in top_segments),
        )
        should_log = (
            self._step_idx % self._cfg.print_every_steps == 0
            or contact_count > 0
            or contact_detected
            or signature != self._last_logged_signature
        )
        if should_log:
            print(
                "[force-debug] "
                f"step={self._step_idx} contact={contact_count} "
                f"det={int(contact_detected)} "
                f"avail={int(available)} Ftot={self._format_force(total_norm)} "
                f"Fwire={self._format_force(wire_norm)} "
                f"Fcoll={self._format_force(collision_norm)} "
                f"Fsum={self._format_force(force_norm_sum)} "
                f"LCPmax={self._format_force(lcp_max_abs)} "
                f"source={source} chan={force_channel} "
                f"wireSrc={wire_force_src} collSrc={coll_force_src} "
                f"top={top_segments_str}"
                + (f" err={error}" if error else ""),
                flush=True,
            )
            self._last_logged_signature = signature

    def reset(self, episode_nr: int = 0) -> None:
        super().reset(episode_nr)
        self._step_idx = 0
        self._last_logged_signature = None
        self._update_debug_outputs()

    def render(self) -> np.ndarray:
        image = super().render()
        self._step_idx += 1
        self._update_debug_outputs()
        return image
