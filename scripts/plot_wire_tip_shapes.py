#!/usr/bin/env python3
"""Export a thesis-ready comparison figure of the registered wire tip shapes."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# Mirrored from data/wire_registry/amplatz_super_stiff/wire_versions/*/tool.py.
# Keeping this script self-contained avoids requiring SOFA/stEVE imports for a
# static thesis figure.
BASELINE_TIP_LENGTH_MM = 15.205308443374598
STRAIGHT_TIP_RADIUS_MM = 1000.0
GENTLE_TIP_RADIUS_MM = 25.0
BASELINE_TIP_RADIUS_MM = 12.1
TIGHT_J_TIP_RADIUS_MM = 8.0
STRONG_HOOK_TIP_RADIUS_MM = 6.0


@dataclass(frozen=True)
class TipVariant:
    label: str
    radius_mm: float
    color: str


TIP_VARIANTS = (
    TipVariant("Straight", STRAIGHT_TIP_RADIUS_MM, "#2f3437"),
    TipVariant("Gentle", GENTLE_TIP_RADIUS_MM, "#2f7d66"),
    TipVariant("Standard J", BASELINE_TIP_RADIUS_MM, "#1f5aa6"),
    TipVariant("Tight J", TIGHT_J_TIP_RADIUS_MM, "#b45f06"),
    TipVariant("Strong hook", STRONG_HOOK_TIP_RADIUS_MM, "#9b2226"),
)


def _tip_centerline(radius_mm: float, n: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Return distal tip centerline coordinates in millimetres.

    The registered tools keep distal arc length fixed and vary curvature through
    the arc radius. Coordinates are oriented with the proximal end at x=0 and
    the tip extending to the right.
    """
    arc_length_mm = BASELINE_TIP_LENGTH_MM
    theta = arc_length_mm / radius_mm
    t = np.linspace(0.0, theta, n)
    x = radius_mm * np.sin(t)
    y = radius_mm * (1.0 - np.cos(t))
    return x, y


def _draw_scale_bar(ax: plt.Axes, length_mm: float = 5.0) -> None:
    y = -2.0
    ax.plot([0.0, length_mm], [y, y], color="#222222", lw=1.1)
    ax.text(
        length_mm / 2.0,
        y - 0.75,
        f"{length_mm:g} mm",
        ha="center",
        va="top",
        fontsize=8,
        color="#222222",
    )


def plot_tip_shapes(
    variants: Iterable[TipVariant],
    output: Path,
    *,
    format_label: str | None = None,
) -> None:
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(10.5, 2.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for ax, variant in zip(axes, variants):
        x, y = _tip_centerline(variant.radius_mm)

        ax.set_facecolor("white")
        ax.plot(
            x,
            y,
            color=variant.color,
            lw=3.0,
            solid_capstyle="round",
        )
        ax.scatter([x[-1]], [y[-1]], s=24, color=variant.color, zorder=3)
        ax.set_title(variant.label, fontsize=10, pad=7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.8, 16.5)
        ax.set_ylim(-3.2, 15.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        angle_deg = math.degrees(BASELINE_TIP_LENGTH_MM / variant.radius_mm)
        ax.text(
            0.0,
            14.4,
            f"r = {variant.radius_mm:g} mm\narc = {BASELINE_TIP_LENGTH_MM:.1f} mm\nangle = {angle_deg:.0f} deg",
            ha="left",
            va="top",
            fontsize=7.5,
            color="#444444",
        )

    _draw_scale_bar(axes[0])

    if format_label:
        fig.suptitle(format_label, fontsize=11, y=1.04)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _tube_mesh(
    x: np.ndarray,
    z: np.ndarray,
    *,
    y_offset: float,
    tube_radius_mm: float = 0.22,
    sides: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.stack(
        [x, np.full_like(x, y_offset, dtype=float), z + tube_radius_mm], axis=1
    )
    tangent = np.gradient(center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    binormal = np.tile(np.array([0.0, 1.0, 0.0]), (center.shape[0], 1))
    normal = np.cross(binormal, tangent)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=True)
    circle = (
        np.cos(angles)[None, :, None] * binormal[:, None, :]
        + np.sin(angles)[None, :, None] * normal[:, None, :]
    )
    tube = center[:, None, :] + tube_radius_mm * circle
    return tube[:, :, 0], tube[:, :, 1], tube[:, :, 2]


def _shade_color(hex_color: str, shade: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(hex_color), dtype=float)
    return tuple(np.clip(rgb * shade + (1.0 - shade), 0.0, 1.0))


def plot_tip_scene(
    variants: Iterable[TipVariant],
    output: Path,
    *,
    format_label: str | None = None,
) -> None:
    variants = tuple(variants)
    fig = plt.figure(figsize=(10.5, 3.1))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass

    x_offsets = np.arange(len(variants), dtype=float) * 18.0
    y_offset = 0.0

    for idx, (variant, x_offset) in enumerate(zip(variants, x_offsets)):
        x, z = _tip_centerline(variant.radius_mm, n=220)
        x = x + x_offset
        xs, ys, zs = _tube_mesh(x, z, y_offset=y_offset)
        ax.plot(
            x,
            np.full_like(x, y_offset + 0.55),
            np.full_like(x, -0.06),
            color="#d8d8d8",
            lw=5.0,
            alpha=0.8,
            solid_capstyle="round",
        )
        ax.plot_surface(
            xs,
            ys,
            zs,
            color=_shade_color(variant.color, 0.92),
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        ax.scatter(
            [x[-1]],
            [y_offset],
            [z[-1] + 0.22],
            s=34,
            color=variant.color,
            depthshade=True,
        )
        fig.text(
            0.12 + idx * 0.18,
            0.16,
            variant.label,
            ha="center",
            va="center",
            fontsize=11,
            color="#222222",
        )

    ax.plot([0.0, 5.0], [1.5, 1.5], [-0.25, -0.25], color="#222222", lw=1.4)
    fig.text(0.115, 0.08, "5 mm", ha="center", va="center", fontsize=8, color="#222222")

    ax.set_xlim(-1.0, x_offsets[-1] + 16.8)
    ax.set_ylim(-1.8, 2.4)
    ax.set_zlim(-1.4, 13.0)
    ax.view_init(elev=7, azim=-90)
    ax.set_box_aspect((x_offsets[-1] + 17.8, 4.2, 22.0))
    ax.set_axis_off()

    if format_label:
        fig.suptitle(format_label, fontsize=11, y=0.95)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.05)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the five Amplatz Super Stiff tip-shape variants."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/wire_tip_shapes.png"),
        help="Output figure path. Use .pdf or .svg for vector output.",
    )
    parser.add_argument(
        "--mode",
        choices=("panel", "scene"),
        default="panel",
        help="panel draws orthographic 2D subplots; scene draws a 3D studio layout.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Omitted by default for thesis layouts.",
    )
    args = parser.parse_args()

    if args.mode == "scene":
        plot_tip_scene(TIP_VARIANTS, args.output, format_label=args.title)
    else:
        plot_tip_shapes(TIP_VARIANTS, args.output, format_label=args.title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
