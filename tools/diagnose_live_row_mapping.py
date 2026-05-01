#!/usr/bin/env python3
"""Replay one eval_v2 trace seed live and dump row-anchored contact mappings."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from steve_recommender.eval_v2.cli import _parse_wire_ref, _parse_branch_names
from steve_recommender.eval_v2.force_telemetry import (
    EvalV2ForceTelemetryCollector,
    _project_constraint_forces,
    _read_data_field,
    _to_float,
)
from steve_recommender.eval_v2.models import (
    BranchEndTarget,
    EvaluationCandidate,
    EvaluationScenario,
    ExecutionPlan,
    FluoroscopySpec,
    ForceTelemetrySpec,
)
from steve_recommender.eval_v2.runner import (
    _flatten_observation,
    _reset_play_policy,
    _reset_single_trial_env,
    _select_action,
    _to_env_action,
    build_single_trial_env,
)
from steve_recommender.eval_v2.runtime import prepare_evaluation_runtime
from steve_recommender.eval_v2.service import DefaultEvaluationService


CENTROID_SEPARATION_FLAG_MM = 5.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a trace seed live and inspect row-to-triangle mappings.",
    )
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--execution-wire",
        default="steve_default/standard_j",
        help="Execution wire ref formatted as model/wire.",
    )
    parser.add_argument(
        "--policy-name",
        default="archvar_original_best",
        help="Policy name used for the original trace run.",
    )
    parser.add_argument(
        "--target-branches",
        default="lcca",
        help="Comma-separated branch list for branch_end targets.",
    )
    parser.add_argument("--policy-device", default="cpu")
    return parser


def _sorted_unique_ints(values: set[int]) -> list[int]:
    return sorted(int(value) for value in values)


def main() -> int:
    args = build_parser().parse_args()
    service = DefaultEvaluationService()

    from steve_recommender.eval_v2.force_trace_persistence import TraceReader

    with TraceReader(args.trace_path) as trace_reader:
        scenario_attrs = dict(trace_reader._file["scenario"].attrs)
        anatomy = service.get_anatomy(record_id=str(scenario_attrs["anatomy_id"]))
        execution_wire = _parse_wire_ref(args.execution_wire)
        policies = service.list_registry_policies(execution_wire=execution_wire) + service.list_explicit_policies(
            execution_wire=execution_wire
        )
        policy = next(
            candidate_policy
            for candidate_policy in policies
            if candidate_policy.name == args.policy_name
        )
        candidate = service.build_candidate(
            name=args.policy_name,
            execution_wire=execution_wire,
            policy=policy,
        )
        scenario = EvaluationScenario(
            name="diag",
            anatomy=anatomy,
            target=BranchEndTarget(branches=_parse_branch_names(args.target_branches)),
            fluoroscopy=FluoroscopySpec(image_frequency_hz=7.5, image_rot_zx_deg=(20.0, 5.0)),
            friction=float(scenario_attrs["friction_mu"]),
            force_telemetry=ForceTelemetrySpec(
                tip_threshold_mm=float(scenario_attrs["tip_threshold_mm"]),
                write_full_trace=False,
                write_diagnostics=False,
            ),
        )
        execution = ExecutionPlan(
            trials_per_candidate=1,
            base_seed=int(scenario_attrs["env_seed"]),
            explicit_seeds=(),
            policy_base_seed=1000,
            policy_explicit_seeds=(),
            max_episode_steps=int(scenario_attrs["max_episode_steps"]),
            policy_device=str(args.policy_device),
            policy_mode="deterministic",
            stochastic_environment_mode="random_start",
            worker_count=1,
        )

    runtime = prepare_evaluation_runtime(
        candidate=candidate,
        scenario=scenario,
        policy_device=args.policy_device,
    )
    env = build_single_trial_env(
        runtime,
        max_episode_steps=execution.max_episode_steps,
        visualisation=None,
    )
    try:
        observation, _ = _reset_single_trial_env(
            env,
            seed=int(scenario_attrs["env_seed"]),
        )
        _reset_play_policy(runtime.play_policy)
        collector = EvalV2ForceTelemetryCollector(
            spec=runtime.scenario.force_telemetry,
            action_dt_s=runtime.scenario.action_dt_s,
        )
        collector.ensure_runtime(intervention=runtime.intervention)

        for step_index in range(args.step + 1):
            flat_state = _flatten_observation(observation)
            action = _select_action(
                runtime,
                flat_state=flat_state,
                execution=execution,
            )
            env_action = _to_env_action(
                action,
                env=env,
                normalize_action=runtime.scenario.normalize_action,
            )
            observation, _, terminated, truncated, _ = env.step(env_action)
            collector.capture_step(intervention=runtime.intervention, step_index=step_index + 1)
            if terminated or truncated:
                raise RuntimeError(f"Trial ended before requested step {args.step}")

        root = runtime.intervention.simulation.root
        export = getattr(root, "wire_wall_contact_export", None)
        if export is None:
            raise RuntimeError("wire_wall_contact_export missing on live scene")

        lcp = getattr(root, "LCP", None)
        if lcp is None:
            raise RuntimeError("LCP object missing on live scene")

        raw_lcp = _read_data_field(lcp, "constraintForces", None)
        lcp_arr = np.asarray(raw_lcp, dtype=np.float64).reshape(-1)
        collision_obj = getattr(root, "InstrumentCombined").CollisionModel.CollisionDOFs
        constraint_raw = _read_data_field(collision_obj, "constraint", None)
        positions = np.asarray(_read_data_field(collision_obj, "position", None), dtype=np.float64).reshape((-1, 3))
        dt_s = _to_float(_read_data_field(root, "dt", None))
        if dt_s is None or dt_s <= 0.0:
            dt_s = runtime.scenario.action_dt_s

        _, row_contribs = _project_constraint_forces(
            lcp_forces=lcp_arr,
            constraint_raw=constraint_raw,
            n_points=int(positions.shape[0]),
            dt_s=dt_s,
        )
        nonzero_rows = {
            int(record["row_idx"])
            for record in row_contribs
            if float(record["force_norm"]) > 0.0
        }

        row_indices = np.asarray(_read_data_field(export, "constraintRowIndices", None), dtype=np.int32).reshape(-1)
        row_valid = np.asarray(_read_data_field(export, "constraintRowValidFlags", None), dtype=np.uint32).reshape(-1)
        wall_ids = np.asarray(_read_data_field(export, "wallTriangleIds", None), dtype=np.int32).reshape(-1)
        triangle_valid = np.asarray(_read_data_field(export, "triangleIdValidFlags", None), dtype=np.uint32).reshape(-1)
        collision_dofs = np.asarray(_read_data_field(export, "collisionDofIndices", None), dtype=np.int32).reshape(-1)
        collision_valid = np.asarray(_read_data_field(export, "collisionDofValidFlags", None), dtype=np.uint32).reshape(-1)

        row_to_tri: dict[int, set[int]] = defaultdict(set)
        row_to_cdof: dict[int, set[int]] = defaultdict(set)
        row_to_export_indices: dict[int, list[int]] = defaultdict(list)
        n_records = min(
            row_indices.size,
            row_valid.size,
            wall_ids.size,
            triangle_valid.size,
            collision_dofs.size,
            collision_valid.size,
        )
        for i in range(n_records):
            row = int(row_indices[i])
            if not bool(int(row_valid[i])) or row < 0:
                continue
            row_to_export_indices[row].append(i)
            if bool(int(triangle_valid[i])):
                row_to_tri[row].add(int(wall_ids[i]))
            if bool(int(collision_valid[i])):
                row_to_cdof[row].add(int(collision_dofs[i]))

        anatomy_mesh = pv.read(
            str(
                Path("data/anatomy_registry/anatomies")
                / str(scenario_attrs["anatomy_id"])
                / "mesh"
                / "simulationmesh.obj"
            )
        )
        centroids = np.asarray(anatomy_mesh.cell_centers().points, dtype=np.float64)

        print(f"trace={args.trace_path}")
        print(f"step={args.step}")
        print(f"nonzero_rows={sorted(nonzero_rows)}")
        print("")

        flagged_any = False
        for row in sorted(nonzero_rows):
            triangles = _sorted_unique_ints(row_to_tri.get(row, set()))
            cdofs = _sorted_unique_ints(row_to_cdof.get(row, set()))
            print(f"row={row}")
            print(f"  triangles={triangles}")
            print(f"  collision_dofs={cdofs}")
            for cdof in cdofs:
                print(
                    "  dof_position_mm"
                    f"[{cdof}]={[round(float(v), 3) for v in positions[cdof].tolist()]}"
                )
            for tri in triangles:
                print(
                    "  triangle_centroid_mm"
                    f"[{tri}]={[round(float(v), 3) for v in centroids[tri].tolist()]}"
                )
            for tri in triangles:
                for cdof in cdofs:
                    dist_mm = float(np.linalg.norm(centroids[tri] - positions[cdof]))
                    print(f"  tri_dof_distance_mm[{tri},{cdof}]={dist_mm:.3f}")

            if len(triangles) > 1:
                max_sep_mm = 0.0
                for i, tri_a in enumerate(triangles):
                    for tri_b in triangles[i + 1 :]:
                        sep_mm = float(np.linalg.norm(centroids[tri_a] - centroids[tri_b]))
                        max_sep_mm = max(max_sep_mm, sep_mm)
                if max_sep_mm > CENTROID_SEPARATION_FLAG_MM:
                    flagged_any = True
                    print(
                        f"  FLAG multi_triangle_row centroid_separation_mm={max_sep_mm:.3f}"
                    )
                    print("  export_indices=" + str(row_to_export_indices.get(row, [])))
                    for i in row_to_export_indices.get(row, []):
                        print(
                            "    export_record"
                            f"[{i}]={{row:{int(row_indices[i])}, row_valid:{int(row_valid[i])}, "
                            f"triangle:{int(wall_ids[i])}, triangle_valid:{int(triangle_valid[i])}, "
                            f"collision_dof:{int(collision_dofs[i])}, collision_valid:{int(collision_valid[i])}}}"
                        )
            print("")

        if not flagged_any:
            print("No flagged multi-triangle rows with centroid separation > 5 mm.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            runtime.play_policy.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
