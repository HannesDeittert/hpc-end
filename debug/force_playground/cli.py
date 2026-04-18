from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .config import ControlConfig, ForcePlaygroundConfig, MeshConfig, OracleConfig, parse_units
from .controllers import build_controller
from .io import PlaygroundRunIO
from .oracle import NormalForceBalanceOracle
from .plotting import LivePlotter
from .report import build_summary_markdown
from .scene_factory import build_scene
from .telemetry import TelemetryCollector
from .viewer import SofaSceneViewer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Isolated force playground for controlled SOFA wall-contact debugging. "
            "Runs outside the production eval/train/comparison pipeline."
        )
    )
    p.add_argument("--scene", choices=["plane_wall", "tube_wall"], default="plane_wall")
    p.add_argument("--probe", choices=["rigid_probe", "guidewire"], default="rigid_probe")
    p.add_argument("--mode", choices=["displacement", "open_loop_force"], default="displacement")
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--alarm-distance", type=float, default=0.5)
    p.add_argument("--contact-distance", type=float, default=0.3)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--plugin-path", default=None)
    p.add_argument("--units", default="mm,kg,s", help="Units as '<length>,<mass>,<time>'")

    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default=None)

    p.add_argument("--plot", dest="plot", action="store_true")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.set_defaults(plot=True)

    p.add_argument("--show-sofa", dest="show_sofa", action="store_true")
    p.add_argument("--no-show-sofa", dest="show_sofa", action="store_false")
    p.set_defaults(show_sofa=False)
    p.add_argument(
        "--camera-preset",
        choices=["auto", "plane_front", "plane_oblique", "tube_oblique"],
        default="auto",
    )

    p.add_argument("--interactive", dest="interactive", action="store_true")
    p.add_argument("--non-interactive", dest="interactive", action="store_false")
    p.set_defaults(interactive=True)

    p.add_argument("--save-plot-snapshots", action="store_true")

    p.add_argument("--insert-action", type=float, default=0.2)
    p.add_argument("--rotate-action", type=float, default=0.0)
    p.add_argument("--open-loop-force-n", type=float, default=0.10)
    p.add_argument("--open-loop-force-node-index", type=int, default=-1)
    p.add_argument("--open-loop-insert-action", type=float, default=0.0)
    p.add_argument("--action-step-delta", type=float, default=0.05)
    p.add_argument("--force-step-delta-n", type=float, default=0.01)

    p.add_argument("--oracle", dest="oracle_enabled", action="store_true")
    p.add_argument("--no-oracle", dest="oracle_enabled", action="store_false")
    p.set_defaults(oracle_enabled=True)
    p.add_argument("--oracle-rel-tol", type=float, default=0.10)
    p.add_argument("--oracle-abs-tol-n", type=float, default=0.01)
    p.add_argument("--oracle-near-zero-ref-n", type=float, default=0.02)
    p.add_argument("--oracle-warmup-steps", type=int, default=40)
    p.add_argument("--oracle-window-steps", type=int, default=80)
    p.add_argument("--require-oracle-applicable", action="store_true")
    p.add_argument("--require-oracle-pass", action="store_true")

    p.add_argument("--plane-width-mm", type=float, default=220.0)
    p.add_argument("--plane-height-mm", type=float, default=120.0)
    p.add_argument("--tube-radius-mm", type=float, default=0.6)
    p.add_argument("--tube-length-mm", type=float, default=50.0)
    p.add_argument("--tube-segments", type=int, default=12)
    p.add_argument("--tube-rings", type=int, default=24)

    return p.parse_args()


def _make_config(args: argparse.Namespace) -> ForcePlaygroundConfig:
    require_oracle_pass = bool(args.require_oracle_pass)
    require_oracle_applicable = bool(args.require_oracle_applicable) or require_oracle_pass
    cfg = ForcePlaygroundConfig(
        scene=args.scene,
        probe=args.probe,
        mode=args.mode,
        tool_ref=str(args.tool_ref),
        steps=int(args.steps),
        seed=int(args.seed),
        friction=float(args.friction),
        image_frequency_hz=float(args.image_frequency_hz),
        alarm_distance=float(args.alarm_distance),
        contact_distance=float(args.contact_distance),
        contact_epsilon=float(args.contact_epsilon),
        plugin_path=args.plugin_path,
        units=parse_units(args.units),
        interactive=bool(args.interactive),
        plot=bool(args.plot),
        show_sofa=bool(args.show_sofa),
        camera_preset=str(args.camera_preset),
        save_plot_snapshots=bool(args.save_plot_snapshots),
        require_oracle_applicable=require_oracle_applicable,
        require_oracle_pass=require_oracle_pass,
        output_root=str(args.output_root),
        run_name=args.run_name,
        mesh=MeshConfig(
            plane_width_mm=float(args.plane_width_mm),
            plane_height_mm=float(args.plane_height_mm),
            tube_radius_mm=float(args.tube_radius_mm),
            tube_length_mm=float(args.tube_length_mm),
            tube_segments=int(args.tube_segments),
            tube_rings=int(args.tube_rings),
        ),
        control=ControlConfig(
            insert_action=float(args.insert_action),
            rotate_action=float(args.rotate_action),
            open_loop_force_n=float(args.open_loop_force_n),
            open_loop_force_node_index=int(args.open_loop_force_node_index),
            open_loop_insert_action=float(args.open_loop_insert_action),
            action_step_delta=float(args.action_step_delta),
            force_step_delta_n=float(args.force_step_delta_n),
        ),
        oracle=OracleConfig(
            enabled=bool(args.oracle_enabled),
            rel_tol=float(args.oracle_rel_tol),
            abs_tol_n=float(args.oracle_abs_tol_n),
            near_zero_ref_n=float(args.oracle_near_zero_ref_n),
            warmup_steps=int(args.oracle_warmup_steps),
            window_steps=int(args.oracle_window_steps),
        ),
    )
    if cfg.interactive and not cfg.plot:
        cfg = replace(cfg, interactive=False)
        print("[force-playground] interactive requested without plot; switching to non-interactive run.")
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = _make_config(args)

    run_dir = PlaygroundRunIO.build_run_dir(cfg.output_root, cfg.effective_run_name())
    io = PlaygroundRunIO(run_dir)
    io.write_config(cfg.to_dict())

    scene = build_scene(cfg, run_dir)
    controller = build_controller(cfg, scene.wall_reference_normal)

    snapshot_dir: Optional[Path] = None
    if cfg.save_plot_snapshots:
        snapshot_dir = run_dir / "plots"

    plotter = LivePlotter(
        enabled=cfg.plot,
        interactive=cfg.interactive,
        snapshot_dir=snapshot_dir,
        on_adjust_key=controller.on_key,
    )

    viewer = SofaSceneViewer(
        enabled=cfg.show_sofa,
        intervention=scene.intervention,
        reset_seed=cfg.seed,
        camera_position=scene.camera_position,
        camera_look_at=scene.camera_look_at,
    )
    if cfg.show_sofa and viewer.should_quit:
        print(
            "[force-playground] warning: failed to initialize sofa window, continuing without scene view."
        )
        if viewer.last_error:
            print(f"[force-playground] viewer_error: {viewer.last_error}")
        viewer.enabled = False
        viewer.should_quit = False
    telemetry = TelemetryCollector(scene, cfg)
    oracle = NormalForceBalanceOracle(cfg, scene.wall_reference_normal)

    print(f"[force-playground] run_dir={run_dir}")
    print(
        "[force-playground] scene={scene} probe={probe} mode={mode} steps={steps}"
        .format(scene=cfg.scene, probe=cfg.probe, mode=cfg.mode, steps=cfg.steps)
    )
    print(
        "[force-playground] camera_preset requested={requested} resolved={resolved}"
        .format(requested=cfg.camera_preset, resolved=scene.camera_preset)
    )
    if cfg.plot and cfg.interactive:
        print("[force-playground] hotkeys: space/n step | c run/pause | q quit | p snapshot | up/down insert | +/- force")
    if cfg.show_sofa and viewer.enabled:
        print("[force-playground] sofa window enabled (camera: arrows + w/s zoom)")
        cam_state = viewer.camera_state()
        print(
            "[force-playground] camera position={pos} look_at={look}"
            .format(pos=cam_state.get("position"), look=cam_state.get("look_at"))
        )
    elif cfg.show_sofa and not viewer.enabled:
        print("[force-playground] sofa window disabled after init failure")
    print(
        "[force-playground] oracle applicable={app} reason={reason}"
        .format(
            app=oracle.applicable,
            reason=(oracle.applicability_reason or "ok"),
        )
    )

    abort_reason = ""
    if cfg.require_oracle_applicable and not oracle.applicable:
        abort_reason = (
            "required oracle is not applicable for this run: "
            + (oracle.applicability_reason or "unknown_reason")
        )
        print(f"[force-playground] ERROR: {abort_reason}")

    idle_callback = viewer.poll_events if viewer.enabled else None
    step_records = []
    oracle_report = {}
    try:
        if not abort_reason:
            for step in range(1, int(cfg.steps) + 1):
                if not plotter.wait_until_step_allowed(idle_callback=idle_callback):
                    break
                if plotter.should_quit:
                    break

                cmd = controller.command(step)
                payload = telemetry.step(step, cmd)
                step_record = oracle.evaluate_step(payload.step_record)

                io.append_step(step_record)
                io.append_triangles(payload.triangle_rows)
                step_records.append(step_record)
                plotter.update(step_record)
                viewer.render_frame()

                if plotter.should_quit or viewer.should_quit:
                    break
    except KeyboardInterrupt:
        print("[force-playground] interrupted by user")
    finally:
        telemetry.close()
        plotter.close()
        viewer.close()

        oracle_report = oracle.finalize()
        io.write_oracle_report(oracle_report)
        summary = build_summary_markdown(cfg, step_records, oracle_report)
        io.write_summary(summary)
        io.close()

    print(f"[force-playground] steps_recorded={len(step_records)}")
    print(f"[force-playground] config: {run_dir / 'config.json'}")
    print(f"[force-playground] steps.csv: {run_dir / 'steps.csv'}")
    print(f"[force-playground] steps.jsonl: {run_dir / 'steps.jsonl'}")
    print(f"[force-playground] triangle_forces.csv: {run_dir / 'triangle_forces.csv'}")
    print(f"[force-playground] oracle_report.json: {run_dir / 'oracle_report.json'}")
    print(f"[force-playground] summary.md: {run_dir / 'summary.md'}")

    if cfg.require_oracle_pass and bool(oracle_report.get("passed")) is not True:
        detail = oracle_report.get("window", {})
        msg = f"required oracle pass failed: passed={oracle_report.get('passed')} window={detail}"
        print(f"[force-playground] ERROR: {msg}")
        raise SystemExit(2)
    if abort_reason:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
