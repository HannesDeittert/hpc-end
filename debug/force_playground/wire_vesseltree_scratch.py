from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from steve_recommender.adapters import eve, eve_rl
from steve_recommender.devices import make_device
from steve_recommender.evaluation.config import ForceUnitsConfig
from steve_recommender.evaluation.force_si import unit_scale_to_si_newton
from steve_recommender.evaluation.info_collectors import SofaWallForceInfo
from steve_recommender.evaluation.torch_checkpoint_compat import (
    legacy_checkpoint_load_context,
)
from steve_recommender.storage import parse_wire_ref, wire_agents_dir
from steve_recommender.visualisation.interactive_sofapygame import InteractiveSofaPygame


_CHECKPOINT_NUM_RE = re.compile(r"^checkpoint(?P<step>\d+)\.everl$")


def _read_data(obj: Any, name: str) -> Any:
    try:
        data = getattr(obj, name)
    except Exception:
        return None
    if hasattr(data, "value"):
        try:
            return data.value
        except Exception:
            return None
    return data


def _set_data_field(obj: Any, name: str, value: Any) -> bool:
    try:
        data = getattr(obj, name)
        if hasattr(data, "value"):
            data.value = value
        else:
            setattr(obj, name, value)
        return True
    except Exception:
        pass
    try:
        data = obj.findData(name)
        data.value = value
        return True
    except Exception:
        return False


def _normalize_xyz_array(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0 or arr.ndim == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % 6 == 0:
            arr = arr.reshape((-1, 6))[:, :3]
        elif arr.size % 3 == 0:
            arr = arr.reshape((-1, 3))
        else:
            return np.zeros((0, 3), dtype=np.float64)
    elif arr.ndim >= 2:
        if arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float64)
        arr = arr[:, :3]
    return np.asarray(arr, dtype=np.float64)


def _parse_constraint_rows(raw: Any) -> List[Tuple[int, int, np.ndarray]]:
    text = str(raw or "").strip()
    if not text:
        return []

    entries: List[Tuple[int, int, np.ndarray]] = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            row_idx = int(float(toks[0]))
            n_blocks = int(float(toks[1]))
        except Exception:
            continue
        if n_blocks <= 0:
            continue
        payload = toks[2:]
        if not payload or len(payload) % n_blocks != 0:
            continue
        block_width = len(payload) // n_blocks
        if block_width < 4:
            continue
        for bi in range(n_blocks):
            base = bi * block_width
            block = payload[base : base + block_width]
            try:
                dof_idx = int(float(block[0]))
                coeff = np.asarray(
                    [float(block[1]), float(block[2]), float(block[3])],
                    dtype=np.float64,
                )
            except Exception:
                continue
            if not np.all(np.isfinite(coeff)):
                continue
            entries.append((row_idx, dof_idx, coeff))
    return entries


def _project_lambda_dt_to_dofs(
    *,
    lambda_dt: np.ndarray,
    row_entries: List[Tuple[int, int, np.ndarray]],
    n_dofs: int,
) -> np.ndarray:
    if int(n_dofs) <= 0 or lambda_dt.size == 0 or not row_entries:
        return np.zeros((0, 3), dtype=np.float64)
    per_dof = np.zeros((int(n_dofs), 3), dtype=np.float64)
    for row_idx, dof_idx, coeff in row_entries:
        if row_idx < 0 or row_idx >= lambda_dt.shape[0]:
            continue
        if dof_idx < 0 or dof_idx >= per_dof.shape[0]:
            continue
        per_dof[dof_idx] += coeff * float(lambda_dt[row_idx])
    return per_dof


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Standalone wire + vessel-tree SOFA scene built from scratch. "
            "Prints contact lambda and wall-force channels each step."
        )
    )
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument("--arch-type", choices=["I", "II", "IV", "V", "VI", "VII"], default="I")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--dt-simulation", type=float, default=0.006)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--image-rot-z-deg", type=float, default=25.0)
    p.add_argument("--image-rot-x-deg", type=float, default=0.0)
    p.add_argument("--target-threshold-mm", type=float, default=3.0)
    p.add_argument(
        "--target-branches",
        default="",
        help="Comma-separated branch names (empty uses all branch ends).",
    )

    p.add_argument("--insert-action", type=float, default=0.2)
    p.add_argument("--rotate-action", type=float, default=0.0)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--alarm-distance", type=float, default=0.5)
    p.add_argument("--contact-distance", type=float, default=0.3)
    p.add_argument(
        "--force-mode",
        choices=["intrusive_lcp", "passive", "constraint_projected_si_validated"],
        default="intrusive_lcp",
    )
    p.add_argument("--plugin-path", default=None)

    p.add_argument("--gui", action="store_true")
    p.add_argument("--display-width", type=int, default=960)
    p.add_argument("--display-height", type=int, default=720)
    p.add_argument("--manual-control", action="store_true")
    p.add_argument(
        "--agent-ref",
        default="",
        help="Optional pinned agent registry ref: model/wire:agent",
    )
    p.add_argument("--agent-checkpoint", default="")
    p.add_argument("--agent-device", default="cpu")
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--print-every", type=int, default=1)

    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="wire_vesseltree_scratch")
    return p


def _parse_target_branches(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    branches = [p for p in parts if p]
    return branches or None


def _build_intervention(args: argparse.Namespace) -> tuple[Any, float]:
    arch_type = getattr(eve.intervention.vesseltree.ArchType, str(args.arch_type))
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        arch_type=arch_type,
        seed=int(args.seed),
    )
    device = make_device(str(args.tool_ref))
    simulation = eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter(
        friction=float(args.friction),
        dt_simulation=float(args.dt_simulation),
    )
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=float(args.image_frequency_hz),
        image_rot_zx=[float(args.image_rot_z_deg), float(args.image_rot_x_deg)],
    )
    target = eve.intervention.target.BranchEnd(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=float(args.target_threshold_mm),
        branches=_parse_target_branches(str(args.target_branches)),
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=False,
        normalize_action=True,
    )
    intervention.make_non_mp()
    return intervention, 1.0 / float(args.image_frequency_hz)


def _configure_detection_distances(sim: Any, *, alarm_distance: float, contact_distance: float) -> None:
    lmd = getattr(getattr(sim, "root", None), "localmindistance", None)
    if lmd is None:
        return
    _set_data_field(lmd, "alarmDistance", float(alarm_distance))
    _set_data_field(lmd, "contactDistance", float(contact_distance))


def _try_parse_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_existing_path(path_str: str, *, agent_dir: Optional[Path] = None) -> Optional[Path]:
    candidate = Path(path_str).expanduser()
    paths_to_try: List[Path] = []
    if candidate.is_absolute():
        paths_to_try.append(candidate)
    else:
        if agent_dir is not None:
            paths_to_try.append((agent_dir / candidate).resolve())
        paths_to_try.append(candidate.resolve())

    for p in paths_to_try:
        if p.exists() and p.is_file():
            return p
    return None


def _highest_numeric_checkpoint(agent_dir: Path) -> Optional[Path]:
    numbered: List[Tuple[int, Path]] = []
    for p in agent_dir.glob("checkpoint*.everl"):
        m = _CHECKPOINT_NUM_RE.match(p.name)
        if not m:
            continue
        numbered.append((int(m.group("step")), p))
    if not numbered:
        return None
    return max(numbered, key=lambda t: t[0])[1]


def _latest_everl(agent_dir: Path) -> Optional[Path]:
    all_everl = sorted(
        agent_dir.glob("*.everl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return all_everl[0] if all_everl else None


def _resolve_agent_checkpoint_from_dir(agent_dir: Path) -> tuple[Optional[Path], str]:
    metadata = _try_parse_json(agent_dir / "agent.json")

    preferred = metadata.get("preferred_checkpoint")
    if not preferred:
        preferred = metadata.get("checkpoint")
    if isinstance(preferred, str) and preferred.strip():
        resolved = _resolve_existing_path(preferred, agent_dir=agent_dir)
        if resolved is not None:
            return resolved.resolve(), "agent.json"

    best = agent_dir / "best_checkpoint.everl"
    if best.exists():
        return best.resolve(), "best_checkpoint.everl"

    highest = _highest_numeric_checkpoint(agent_dir)
    if highest is not None:
        return highest.resolve(), "highest_checkpointN"

    latest = _latest_everl(agent_dir)
    if latest is not None:
        return latest.resolve(), "latest_everl"

    return None, "none"


def _select_best_agent_checkpoint(
    *,
    tool_ref: str,
    pinned_agent_ref: str,
) -> tuple[Path, str, str]:
    model, wire = parse_wire_ref(str(tool_ref))
    if not model:
        raise ValueError(
            "tool_ref must be 'model/wire' for automatic agent selection "
            f"(got '{tool_ref}')"
        )
    agents_root = wire_agents_dir(model, wire)
    if not agents_root.exists():
        raise FileNotFoundError(f"No agents directory found: {agents_root}")

    pinned = str(pinned_agent_ref or "").strip()
    if pinned:
        if ":" not in pinned:
            raise ValueError(
                f"Invalid --agent-ref '{pinned}'. Expected format model/wire:agent"
            )
        pinned_tool, pinned_agent = pinned.rsplit(":", 1)
        if pinned_tool.strip() != f"{model}/{wire}" or not pinned_agent.strip():
            raise ValueError(
                f"--agent-ref '{pinned}' does not match tool_ref '{model}/{wire}'"
            )
        agent_dir = agents_root / pinned_agent.strip()
        if not agent_dir.exists():
            raise FileNotFoundError(f"Pinned agent directory not found: {agent_dir}")
        resolved, source = _resolve_agent_checkpoint_from_dir(agent_dir)
        if resolved is None:
            raise FileNotFoundError(f"No checkpoint found for pinned agent: {agent_dir}")
        return resolved, pinned, source

    # Auto policy across registered agents:
    # 1) prefer checkpoints named best_checkpoint.everl
    # 2) then checkpoints from agent.json
    # 3) then by newest mtime
    candidates: List[Tuple[int, float, str, Path, str]] = []
    for child in sorted(agents_root.iterdir()):
        if not child.is_dir():
            continue
        resolved, source = _resolve_agent_checkpoint_from_dir(child)
        if resolved is None:
            continue
        rank = 0
        if resolved.name == "best_checkpoint.everl":
            rank += 20
        if source == "agent.json":
            rank += 10
        lname = child.name.lower()
        if "best" in lname:
            rank += 5
        if "hq" in lname:
            rank += 2
        mtime = float(resolved.stat().st_mtime)
        candidates.append((rank, mtime, child.name, resolved, source))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {agents_root}")

    _rank, _mtime, agent_name, ckpt, source = max(candidates, key=lambda x: (x[0], x[1]))
    _ = (_rank, _mtime)
    return ckpt.resolve(), f"{model}/{wire}:{agent_name}", source


def _configure_solver_and_collision(sim: Any) -> tuple[Any, Any]:
    lcp = getattr(getattr(sim, "root", None), "LCP", None)
    if lcp is not None:
        _set_data_field(lcp, "computeConstraintForces", True)
        _set_data_field(lcp, "build_lcp", True)
    collision_dofs = None
    try:
        collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
    except Exception:
        collision_dofs = None
    return lcp, collision_dofs


def _build_agent_env(
    *,
    intervention: Any,
    force_info: SofaWallForceInfo,
    max_episode_steps: int,
    visualisation: Any,
) -> Any:
    start = eve.start.InsertionPoint(intervention)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

    tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
    tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(tracking, intervention)
    tracking = eve.observation.wrapper.Memory(
        tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
    )
    target_state = eve.observation.Target2D(intervention)
    target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
        target_state, intervention
    )
    last_action = eve.observation.LastAction(intervention)
    last_action = eve.observation.wrapper.Normalize(last_action)

    observation = eve.observation.ObsDict(
        {
            "tracking": tracking,
            "target": target_state,
            "last_action": last_action,
        }
    )

    target_reward = eve.reward.TargetReached(
        intervention,
        factor=1.0,
        final_only_after_all_interim=False,
    )
    step_reward = eve.reward.Step(factor=-0.005)
    path_delta = eve.reward.PathLengthDelta(pathfinder, 0.001)
    reward = eve.reward.Combination([target_reward, path_delta, step_reward])

    terminal = eve.terminal.TargetReached(intervention)
    max_steps = eve.truncation.MaxSteps(int(max_episode_steps))
    vessel_end = eve.truncation.VesselEnd(intervention)
    sim_error = eve.truncation.SimError(intervention)
    truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])

    target_reached = eve.info.TargetReached(intervention, name="success")
    path_ratio = eve.info.PathRatio(pathfinder)
    steps_info = eve.info.Steps()
    trans_speed = eve.info.AverageTranslationSpeed(intervention)
    trajectory_length = eve.info.TrajectoryLength(intervention)
    info = eve.info.Combination(
        [target_reached, path_ratio, steps_info, trans_speed, trajectory_length, force_info]
    )

    return eve.Env(
        intervention=intervention,
        observation=observation,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        start=start,
        pathfinder=pathfinder,
        visualisation=visualisation,
        info=info,
        interim_target=None,
    )


def _extract_step_metrics(
    *,
    sim: Any,
    lcp: Any,
    collision_dofs: Any,
    dt_step_s: float,
    contact_epsilon: float,
    force_scale_to_n: float,
    info: dict,
) -> dict[str, Any]:
    impulses = np.zeros((0,), dtype=np.float64)
    if lcp is not None:
        lcp_raw = _read_data(lcp, "constraintForces")
        if lcp_raw is not None:
            impulses = np.asarray(lcp_raw, dtype=np.float64).reshape((-1,))
    lambda_dt = (
        impulses / float(dt_step_s)
        if impulses.size and float(dt_step_s) > 0.0
        else np.zeros((0,), dtype=np.float64)
    )
    active_rows = int(np.count_nonzero(np.abs(impulses) > float(contact_epsilon)))
    lambda_abs_max = float(np.max(np.abs(impulses))) if impulses.size else 0.0

    proj_wall_N = np.zeros((3,), dtype=np.float64)
    if collision_dofs is not None:
        constraint_raw = _read_data(collision_dofs, "constraint")
        row_entries = _parse_constraint_rows(constraint_raw)
        positions = _normalize_xyz_array(_read_data(collision_dofs, "position"))
        proj_dofs_scene = _project_lambda_dt_to_dofs(
            lambda_dt=lambda_dt,
            row_entries=row_entries,
            n_dofs=int(positions.shape[0]),
        )
        if proj_dofs_scene.size:
            proj_wall_N = (
                -np.sum(proj_dofs_scene, axis=0, dtype=np.float64) * force_scale_to_n
            )

    wall_N = np.asarray(
        info.get("wall_total_force_vector_N", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape((3,))
    sim_time = float(getattr(sim.root.time, "value", float("nan")))
    return {
        "sim_time": sim_time,
        "impulses": impulses,
        "active_rows": active_rows,
        "lambda_abs_max": lambda_abs_max,
        "proj_wall_N": proj_wall_N,
        "proj_wall_norm_N": float(np.linalg.norm(proj_wall_N)),
        "wall_N": wall_N,
        "wall_norm_N": float(np.linalg.norm(wall_N)),
        "contact_detected": bool(info.get("wall_contact_detected", False)),
        "contact_count": int(info.get("wall_contact_count", 0)),
        "quality_tier": str(info.get("wall_force_quality_tier", "unavailable")),
        "force_channel": str(info.get("wall_force_channel", "none")),
    }


def main() -> None:
    args = _build_parser().parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "wire_vesseltree_scratch.csv"

    intervention, dt_step_s = _build_intervention(args)
    agent_control = not bool(args.manual_control)
    agent_checkpoint = str(args.agent_checkpoint or "").strip()
    selected_agent_ref = ""
    selected_agent_source = ""

    if agent_control and not agent_checkpoint:
        ckpt_path, selected_agent_ref, selected_agent_source = _select_best_agent_checkpoint(
            tool_ref=str(args.tool_ref),
            pinned_agent_ref=str(args.agent_ref or ""),
        )
        agent_checkpoint = str(ckpt_path)
    elif agent_control and agent_checkpoint and str(args.agent_ref or "").strip():
        selected_agent_ref = str(args.agent_ref).strip()
        selected_agent_source = "checkpoint_override"
    elif agent_control and agent_checkpoint:
        selected_agent_ref = "(checkpoint path only)"
        selected_agent_source = "checkpoint_override"

    if bool(args.gui):
        sim = intervention.simulation
        sim.init_visual_nodes = True
        sim.display_size = (int(args.display_width), int(args.display_height))
        sim.target_size = float(args.target_threshold_mm)
        sim.interim_target_size = float(args.target_threshold_mm)

    viewer = None
    if bool(args.gui):
        try:
            viewer = InteractiveSofaPygame(
                intervention=intervention,
                display_size=(int(args.display_width), int(args.display_height)),
            )
        except Exception as exc:
            viewer = None
            print(f"[scratch] viewer init failed; continuing headless: {exc}")

    force_units = ForceUnitsConfig(length_unit="mm", mass_unit="kg", time_unit="s")
    force_scale_to_n = float(unit_scale_to_si_newton(force_units))
    force_info = SofaWallForceInfo(
        intervention,
        mode=str(args.force_mode),
        required=False,
        contact_epsilon=float(args.contact_epsilon),
        plugin_path=args.plugin_path,
        units=force_units,
        constraint_dt_s=float(dt_step_s),
    )

    print(f"[scratch] run_dir={run_dir}")
    print(
        "[scratch] arch={arch} tool={tool} steps={steps} force_mode={mode} gui={gui} control={control}".format(
            arch=args.arch_type,
            tool=args.tool_ref,
            steps=int(args.steps),
            mode=args.force_mode,
            gui=bool(viewer is not None),
            control=("agent" if agent_control else "manual"),
        )
    )
    if agent_control:
        print(f"[scratch] agent_checkpoint={agent_checkpoint}")
        if selected_agent_ref:
            print(
                f"[scratch] agent_selected={selected_agent_ref} source={selected_agent_source or 'unknown'}"
            )
    print(f"[scratch] output={out_csv}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "step",
                "sim_time_s",
                "lambda_size",
                "active_rows",
                "lambda_abs_max",
                "proj_wall_fx_N",
                "proj_wall_fy_N",
                "proj_wall_fz_N",
                "proj_wall_norm_N",
                "wall_fx_N",
                "wall_fy_N",
                "wall_fz_N",
                "wall_norm_N",
                "contact_detected",
                "contact_count",
                "force_quality_tier",
                "force_channel",
                "action_insert_cmd",
                "action_rotate_cmd",
                "done",
            ]
        )
        if agent_control:
            import torch

            visualisation = viewer if viewer is not None else eve.visualisation.VisualisationDummy()
            env = _build_agent_env(
                intervention=intervention,
                force_info=force_info,
                max_episode_steps=max(int(args.max_episode_steps), int(args.steps)),
                visualisation=visualisation,
            )
            algo = None
            try:
                with legacy_checkpoint_load_context():
                    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(agent_checkpoint)
                algo.to(torch.device(str(args.agent_device)))
                algo.reset()

                obs, _ = env.reset(seed=int(args.seed))
                obs_flat, _ = eve_rl.util.flatten_obs(obs)
                sim = intervention.simulation
                _configure_detection_distances(
                    sim,
                    alarm_distance=float(args.alarm_distance),
                    contact_distance=float(args.contact_distance),
                )
                lcp, collision_dofs = _configure_solver_and_collision(sim)

                for step in range(int(args.steps)):
                    action_raw = algo.get_eval_action(obs_flat)
                    action_raw = np.asarray(action_raw, dtype=np.float32)
                    env_action = action_raw.reshape(env.action_space.shape)
                    env_action = (env_action + 1.0) / 2.0 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low

                    obs, _reward, terminal, truncation, info = env.step(env_action)
                    obs_flat, _ = eve_rl.util.flatten_obs(obs)
                    m = _extract_step_metrics(
                        sim=sim,
                        lcp=lcp,
                        collision_dofs=collision_dofs,
                        dt_step_s=float(dt_step_s),
                        contact_epsilon=float(args.contact_epsilon),
                        force_scale_to_n=force_scale_to_n,
                        info=info,
                    )
                    done = bool(terminal or truncation)
                    action_flat = np.asarray(env_action, dtype=np.float64).reshape((-1,))
                    action_insert = float(action_flat[0]) if action_flat.size > 0 else float("nan")
                    action_rotate = float(action_flat[1]) if action_flat.size > 1 else float("nan")

                    if int(args.print_every) > 0 and step % int(args.print_every) == 0:
                        if m["impulses"].size:
                            print(
                                f"[scratch] t={m['sim_time']:.3f} lambda(n,t1,t2,...)={m['impulses']}"
                            )
                        else:
                            print(f"[scratch] t={m['sim_time']:.3f} lambda=none")
                        print(
                            "[scratch] projected wall reaction N={w} |norm|={n:.6g}".format(
                                w=m["proj_wall_N"],
                                n=m["proj_wall_norm_N"],
                            )
                        )
                        print(
                            "[scratch] wall_total_force_vector_N={w} |norm|={n:.6g} "
                            "contact={c} count={k} tier={t} channel={ch} done={d}".format(
                                w=m["wall_N"],
                                n=m["wall_norm_N"],
                                c=int(m["contact_detected"]),
                                k=m["contact_count"],
                                t=m["quality_tier"],
                                ch=m["force_channel"],
                                d=int(done),
                            )
                        )

                    writer.writerow(
                        [
                            int(step),
                            m["sim_time"],
                            int(m["impulses"].size),
                            int(m["active_rows"]),
                            float(m["lambda_abs_max"]),
                            float(m["proj_wall_N"][0]),
                            float(m["proj_wall_N"][1]),
                            float(m["proj_wall_N"][2]),
                            float(m["proj_wall_norm_N"]),
                            float(m["wall_N"][0]),
                            float(m["wall_N"][1]),
                            float(m["wall_N"][2]),
                            float(m["wall_norm_N"]),
                            int(m["contact_detected"]),
                            int(m["contact_count"]),
                            str(m["quality_tier"]),
                            str(m["force_channel"]),
                            action_insert,
                            action_rotate,
                            int(done),
                        ]
                    )
                    env.render()
                    if done:
                        print(f"[scratch] episode ended at step={step}")
                        break
            except KeyboardInterrupt:
                print("[scratch] interrupted by user")
            finally:
                if algo is not None:
                    try:
                        algo.close()
                    except Exception:
                        pass
                try:
                    env.close()
                except Exception:
                    pass
        else:
            intervention.reset(int(args.seed))
            sim = intervention.simulation
            _configure_detection_distances(
                sim,
                alarm_distance=float(args.alarm_distance),
                contact_distance=float(args.contact_distance),
            )
            lcp, collision_dofs = _configure_solver_and_collision(sim)
            force_info.reset()
            if viewer is not None:
                try:
                    viewer.reset(episode_nr=0)
                except Exception as exc:
                    print(f"[scratch] viewer reset failed; continuing headless: {exc}")
                    viewer = None

            action = np.asarray(
                [[float(args.insert_action), float(args.rotate_action)]],
                dtype=np.float32,
            )
            try:
                for step in range(int(args.steps)):
                    intervention.step(action)
                    force_info.step()
                    info = force_info.info
                    m = _extract_step_metrics(
                        sim=sim,
                        lcp=lcp,
                        collision_dofs=collision_dofs,
                        dt_step_s=float(dt_step_s),
                        contact_epsilon=float(args.contact_epsilon),
                        force_scale_to_n=force_scale_to_n,
                        info=info,
                    )

                    if int(args.print_every) > 0 and step % int(args.print_every) == 0:
                        if m["impulses"].size:
                            print(
                                f"[scratch] t={m['sim_time']:.3f} lambda(n,t1,t2,...)={m['impulses']}"
                            )
                        else:
                            print(f"[scratch] t={m['sim_time']:.3f} lambda=none")
                        print(
                            "[scratch] projected wall reaction N={w} |norm|={n:.6g}".format(
                                w=m["proj_wall_N"],
                                n=m["proj_wall_norm_N"],
                            )
                        )
                        print(
                            "[scratch] wall_total_force_vector_N={w} |norm|={n:.6g} "
                            "contact={c} count={k} tier={t} channel={ch}".format(
                                w=m["wall_N"],
                                n=m["wall_norm_N"],
                                c=int(m["contact_detected"]),
                                k=m["contact_count"],
                                t=m["quality_tier"],
                                ch=m["force_channel"],
                            )
                        )

                    writer.writerow(
                        [
                            int(step),
                            m["sim_time"],
                            int(m["impulses"].size),
                            int(m["active_rows"]),
                            float(m["lambda_abs_max"]),
                            float(m["proj_wall_N"][0]),
                            float(m["proj_wall_N"][1]),
                            float(m["proj_wall_N"][2]),
                            float(m["proj_wall_norm_N"]),
                            float(m["wall_N"][0]),
                            float(m["wall_N"][1]),
                            float(m["wall_N"][2]),
                            float(m["wall_norm_N"]),
                            int(m["contact_detected"]),
                            int(m["contact_count"]),
                            str(m["quality_tier"]),
                            str(m["force_channel"]),
                            float(args.insert_action),
                            float(args.rotate_action),
                            0,
                        ]
                    )

                    if viewer is not None:
                        viewer.render()
            except KeyboardInterrupt:
                print("[scratch] interrupted by user")
            finally:
                if viewer is not None:
                    try:
                        viewer.close()
                    except Exception:
                        pass

    print(f"[scratch] wrote {out_csv}")


if __name__ == "__main__":
    main()
