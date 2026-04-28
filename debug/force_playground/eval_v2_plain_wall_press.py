from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from steve_recommender.eval_v2.force_telemetry import EvalV2ForceTelemetryCollector
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits

from .config import ControlConfig, ForcePlaygroundConfig
from .controllers import ForceApplicator, OpenLoopForceController
from .scene_factory import build_scene


def _unit_scale_to_newton(units: ForceUnits) -> float:
    length_scale = {"m": 1.0, "mm": 1e-3}[str(units.length_unit)]
    mass_scale = {"kg": 1.0, "g": 1e-3}[str(units.mass_unit)]
    time_scale = {"s": 1.0, "ms": 1e-3}[str(units.time_unit)]
    return float(mass_scale * length_scale / (time_scale * time_scale))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plain-wall open-loop press benchmark that uses eval_v2 force telemetry "
            "collector (passive monitor + LCP/dt fallback)."
        )
    )
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--target-force-n", type=float, default=0.10)
    p.add_argument("--insert-action", type=float, default=0.0)
    p.add_argument("--force-node-index", type=int, default=-1)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument(
        "--force-mode",
        choices=["passive", "constraint_projected_si_validated"],
        default="constraint_projected_si_validated",
    )
    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="eval_v2_plain_wall_press")
    p.add_argument(
        "--drive-mode",
        choices=["external_force", "action_only"],
        default="external_force",
        help=(
            "external_force: inject via wire DOFs externalForce/force field; "
            "action_only: no direct force write (motion/contact-only validation)."
        ),
    )
    p.add_argument(
        "--independent-relerr-threshold",
        type=float,
        default=0.25,
        help=(
            "Relative-error threshold for monitor-vs-LCP independent magnitude agreement."
        ),
    )
    p.add_argument(
        "--independent-epsilon-n",
        type=float,
        default=1e-5,
        help="Minimum magnitude (N) for independent agreement samples.",
    )
    return p.parse_args()


def _build_cfg(args: argparse.Namespace) -> ForcePlaygroundConfig:
    return ForcePlaygroundConfig(
        scene="plane_wall",
        probe="rigid_probe",
        mode="open_loop_force",
        steps=int(args.steps),
        seed=int(args.seed),
        friction=float(args.friction),
        image_frequency_hz=float(args.image_frequency_hz),
        contact_epsilon=float(args.contact_epsilon),
        plot=False,
        interactive=False,
        show_sofa=False,
        output_root=str(args.output_root),
        run_name=str(args.run_name),
        control=ControlConfig(
            insert_action=float(args.insert_action),
            rotate_action=0.0,
            open_loop_force_n=float(args.target_force_n),
            open_loop_force_node_index=int(args.force_node_index),
            open_loop_insert_action=float(args.insert_action),
            action_step_delta=0.0,
            force_step_delta_n=0.0,
        ),
    )


def _read_data_field(obj: Any, name: str, default: Any = None) -> Any:
    try:
        data = getattr(obj, name)
        if hasattr(data, "value"):
            return data.value
        return data
    except Exception:
        pass
    try:
        data = obj.findData(name)
        return data.value
    except Exception:
        return default


def main() -> None:
    args = _parse_args()
    cfg = _build_cfg(args)

    run_tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"{run_tag}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene(cfg, run_dir)
    controller = OpenLoopForceController(cfg, scene.wall_reference_normal)
    force_applicator = ForceApplicator(scene.simulation, node_index=int(args.force_node_index))

    units = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
    force_scale_to_newton = _unit_scale_to_newton(units)
    spec = ForceTelemetrySpec(
        mode=str(args.force_mode),
        required=False,
        contact_epsilon=float(args.contact_epsilon),
        units=units if str(args.force_mode) == "constraint_projected_si_validated" else None,
    )
    collector = EvalV2ForceTelemetryCollector(spec=spec, action_dt_s=float(scene.dt_s))
    status = collector.ensure_runtime(intervention=scene.intervention)
    sim_dt_s = float(_read_data_field(scene.simulation.root, "dt", scene.dt_s) or scene.dt_s)

    print(f"[eval_v2-wall-press] run_dir={run_dir}")
    print(
        "[eval_v2-wall-press] configured={configured} source={source} error={error}".format(
            configured=status.configured,
            source=status.source,
            error=(status.error or ""),
        )
    )
    print(
        "[eval_v2-wall-press] mode={mode} drive_mode={drive_mode} steps={steps} target_force_n={force:.6g} action_dt_s={dt:.6g} sim_dt_s={sim_dt:.6g}".format(
            mode=spec.mode,
            drive_mode=str(args.drive_mode),
            steps=cfg.steps,
            force=float(args.target_force_n),
            dt=float(scene.dt_s),
            sim_dt=float(sim_dt_s),
        )
    )

    rows: list[dict[str, Any]] = []
    measured_series: list[float] = []
    fallback_series: list[float] = []

    for step_idx in range(1, int(cfg.steps) + 1):
        command = controller.command(step_idx)

        cmd_force_n = np.asarray(command.commanded_force_vector_n, dtype=np.float64).reshape((3,))
        cmd_force_scene = cmd_force_n / float(force_scale_to_newton)
        if str(args.drive_mode) == "external_force":
            # stEVE ConstantForceField path behaves as impulse-like per integration
            # substep; multiply by sim dt to realize the requested force magnitude.
            cmd_impulse_scene = cmd_force_scene * float(sim_dt_s)
            apply_status = force_applicator.apply_force_scene(cmd_impulse_scene)
        else:
            apply_status = "skipped:action_only"

        scene.intervention.step(np.asarray(command.action, dtype=np.float32))
        collector.capture_step(intervention=scene.intervention, step_index=step_idx)

        root = scene.simulation.root
        monitor = getattr(root, "wire_wall_force_monitor", None)

        monitor_total = float("nan")
        if monitor is not None:
            raw_total = _read_data_field(monitor, "totalForceNorm", None)
            if raw_total is not None:
                monitor_total = float(raw_total)
                if spec.mode == "constraint_projected_si_validated":
                    monitor_total *= float(force_scale_to_newton)
                measured_series.append(float(monitor_total))

        fallback_lcp = float("nan")
        lcp = getattr(root, "LCP", None)
        if lcp is not None:
            raw_lambda = _read_data_field(lcp, "constraintForces", None)
            if raw_lambda is not None:
                lam = np.asarray(raw_lambda, dtype=np.float64).reshape(-1)
                if lam.size > 0:
                    dt_s = float(_read_data_field(root, "dt", scene.dt_s) or scene.dt_s)
                    if dt_s > 0.0:
                        fallback_lcp = float(np.max(np.abs(lam) / dt_s))
                        if spec.mode == "constraint_projected_si_validated":
                            fallback_lcp *= float(force_scale_to_newton)
                        fallback_series.append(fallback_lcp)

        row = {
            "step": int(step_idx),
            "target_force_n": float(args.target_force_n),
            "commanded_force_norm_n": float(np.linalg.norm(cmd_force_n)),
            "monitor_total_force_norm": monitor_total,
            "fallback_lcp_force_norm": fallback_lcp,
            "apply_status": str(apply_status),
        }
        rows.append(row)

    summary = collector.build_summary()

    tail_window = min(40, len(rows))
    tail_rows = rows[-tail_window:] if tail_window > 0 else []
    tail_monitor = [float(r["monitor_total_force_norm"]) for r in tail_rows if np.isfinite(r["monitor_total_force_norm"])]
    tail_fallback = [float(r["fallback_lcp_force_norm"]) for r in tail_rows if np.isfinite(r["fallback_lcp_force_norm"])]

    measured_tail_mean = float(np.mean(tail_monitor)) if tail_monitor else float("nan")
    fallback_tail_mean = float(np.mean(tail_fallback)) if tail_fallback else float("nan")

    independent_pairs: list[tuple[float, float]] = []
    for row in rows:
        monitor_val = float(row["monitor_total_force_norm"])
        lcp_val = float(row["fallback_lcp_force_norm"])
        if not (np.isfinite(monitor_val) and np.isfinite(lcp_val)):
            continue
        if max(abs(monitor_val), abs(lcp_val)) < float(args.independent_epsilon_n):
            continue
        independent_pairs.append((monitor_val, lcp_val))

    independent_rel_errors = [
        abs(m - l) / max(abs(m), abs(l), float(args.independent_epsilon_n))
        for (m, l) in independent_pairs
    ]
    independent_rel_error_mean = (
        float(np.mean(independent_rel_errors)) if independent_rel_errors else float("nan")
    )
    independent_rel_error_max = (
        float(np.max(independent_rel_errors)) if independent_rel_errors else float("nan")
    )
    independent_agreement_ok = bool(
        independent_rel_errors
        and independent_rel_error_mean <= float(args.independent_relerr_threshold)
    )

    report = {
        "run_dir": str(run_dir),
        "mode": spec.mode,
        "drive_mode": str(args.drive_mode),
        "target_force_n": float(args.target_force_n),
        "steps": int(cfg.steps),
        "dt_s": float(scene.dt_s),
        "collector_summary": {
            "available_for_score": bool(summary.available_for_score),
            "validation_status": str(summary.validation_status),
            "source": str(summary.source),
            "channel": str(summary.channel),
            "total_force_norm_max": summary.total_force_norm_max,
            "total_force_norm_mean": summary.total_force_norm_mean,
            "lcp_max_abs_max": summary.lcp_max_abs_max,
            "lcp_sum_abs_mean": summary.lcp_sum_abs_mean,
            "lcp_mapped_wall_row_count_max": summary.lcp_mapped_wall_row_count_max,
            "lcp_contact_export_coverage": summary.lcp_contact_export_coverage,
        },
        "tail_window_steps": int(tail_window),
        "tail_means": {
            "monitor_total_force_norm": measured_tail_mean,
            "fallback_lcp_force_norm": fallback_tail_mean,
        },
        "independent_magnitude_check": {
            "pair_count": int(len(independent_pairs)),
            "epsilon_n": float(args.independent_epsilon_n),
            "relative_error_threshold": float(args.independent_relerr_threshold),
            "relative_error_mean": independent_rel_error_mean,
            "relative_error_max": independent_rel_error_max,
            "agreement_ok": independent_agreement_ok,
        },
    }

    steps_path = run_dir / "eval_v2_wall_press_steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    report_path = run_dir / "eval_v2_wall_press_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[eval_v2-wall-press] summary={report_path}")
    print(
        "[eval_v2-wall-press] tail_mean monitor={m:.6g} fallback={l:.6g} target={t:.6g}".format(
            m=measured_tail_mean,
            l=fallback_tail_mean,
            t=float(args.target_force_n),
        )
    )
    print(
        "[eval_v2-wall-press] independent_check pairs={pairs} mean_relerr={mean:.6g} max_relerr={mx:.6g} threshold={th:.6g} ok={ok}".format(
            pairs=int(len(independent_pairs)),
            mean=independent_rel_error_mean,
            mx=independent_rel_error_max,
            th=float(args.independent_relerr_threshold),
            ok=independent_agreement_ok,
        )
    )


if __name__ == "__main__":
    main()
