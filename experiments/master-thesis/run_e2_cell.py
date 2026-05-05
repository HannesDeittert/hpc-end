from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HELPER_DIR = SCRIPT_DIR / "notebook_helpers"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from steve_recommender.eval_v2.models import (  # noqa: E402
    BranchEndTarget,
    BranchIndexTarget,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationScenario,
    ExecutionPlan,
    ForceScoringSpec,
    ForceTelemetrySpec,
    FluoroscopySpec,
    ScoringSpec,
    WireRef,
)
from e1 import config_spec  # type: ignore  # noqa: E402
from e2_tinyfat import DEFAULT_E2_ROOT, DEFAULT_FRICTION, DEFAULT_WORKER_COUNT  # type: ignore  # noqa: E402


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one E2 job")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--wires-json", type=Path, default=DEFAULT_E2_ROOT / "metadata" / "wires.json")
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--policy-device", default="cpu")
    return parser.parse_args(argv)


def _parse_wire_ref(text: str) -> WireRef:
    model, wire = str(text).split("/", maxsplit=1)
    return WireRef(model=model, wire=wire)


def make_default_service():
    from steve_recommender.eval_v2.service import DefaultEvaluationService
    return DefaultEvaluationService()


def _row_from_manifest(manifest_path: Path, index: int) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if index < 0 or index >= len(jobs):
        raise IndexError(f"array index {index} out of range for {manifest_path}")
    return dict(jobs[index])


def _load_wires(wires_json: Path, service: Any) -> tuple[WireRef, ...]:
    if wires_json.exists():
        payload = json.loads(wires_json.read_text(encoding="utf-8"))
        wires = []
        for item in payload.get("wires", []):
            tool_ref = str(item.get("tool_ref", ""))
            if "/" in tool_ref:
                wires.append(_parse_wire_ref(tool_ref))
        if wires:
            return tuple(wires)
    return tuple(service.list_startable_wires())


def _build_candidate(*, service: Any, execution_wire: WireRef) -> EvaluationCandidate:
    candidates = service.list_candidates(execution_wire=execution_wire, include_cross_wire=False)
    if not candidates:
        raise ValueError(f"No candidates available for {execution_wire.tool_ref}")
    return candidates[0]


def _build_target(target_spec: dict[str, Any]):
    kind = str(target_spec.get("kind", ""))
    threshold_mm = float(target_spec.get("threshold_mm", 5.0))
    if kind == "branch_end":
        branches = tuple(str(branch) for branch in target_spec.get("branches", (target_spec.get("branch"),)) if branch)
        return BranchEndTarget(threshold_mm=threshold_mm, branches=branches)
    if kind == "branch_index":
        return BranchIndexTarget(
            branch=str(target_spec["branch"]),
            index=int(target_spec.get("index", 0)),
            threshold_mm=threshold_mm,
        )
    raise ValueError(f"unsupported E2 target kind: {kind!r}")


def _build_execution_plan(*, config_id: int, seed_base: int, trial_count: int, max_episode_steps: int, worker_count: int) -> ExecutionPlan:
    spec = config_spec(config_id)
    return ExecutionPlan(
        trials_per_candidate=int(trial_count),
        base_seed=int(seed_base),
        policy_base_seed=int(seed_base) + 100_000,
        max_episode_steps=int(max_episode_steps),
        policy_device="cpu",
        policy_mode=spec["policy_mode"],
        stochastic_environment_mode=spec["stochastic_environment_mode"],
        worker_count=int(worker_count),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.manifest is None or args.array_index is None:
        raise ValueError("--manifest and --array-index are required")
    service = make_default_service()
    row = _row_from_manifest(args.manifest, args.array_index)
    anatomy_id = str(row["anatomy_id"])
    target_spec = dict(row["target_spec"])
    config_id = int(row["config_id"])
    trial_count = int(row["config_spec"]["trial_count"])
    seed_base = int(row["seed_base"])
    output_path = Path(str(row["output_dir"]))
    max_episode_steps = int(row["config_spec"].get("max_episode_steps", 1000))
    write_full_trace = bool(row.get("write_full_trace", False))
    write_diagnostics = bool(row.get("write_diagnostics", False))
    friction = float(row.get("friction", os.environ.get("E2_FRICTION", DEFAULT_FRICTION)))

    anatomy = service.get_anatomy(record_id=anatomy_id)
    scenario = EvaluationScenario(
        name=f"{anatomy_id}__target_{int(target_spec.get('target_index', 0))}",
        anatomy=anatomy,
        target=_build_target(target_spec),
        fluoroscopy=FluoroscopySpec(),
        force_telemetry=ForceTelemetrySpec(
            mode="passive",
            required=False,
            tip_threshold_mm=float(target_spec.get("threshold_mm", 5.0)),
            write_full_trace=write_full_trace,
            write_diagnostics=write_diagnostics,
            plugin_path=None,
            units=None,
        ),
        friction=friction,
    )
    wires = _load_wires(Path(args.wires_json), service)
    candidates = tuple(_build_candidate(service=service, execution_wire=wire) for wire in wires)
    job = EvaluationJob(
        name=output_path.name,
        scenarios=(scenario,),
        candidates=candidates,
        execution=_build_execution_plan(
            config_id=config_id,
            seed_base=seed_base,
            trial_count=trial_count,
            max_episode_steps=max_episode_steps,
            worker_count=int(args.worker_count),
        ),
        scoring=ScoringSpec(force=ForceScoringSpec(tip_length_mm=float(target_spec.get("threshold_mm", 5.0)))),
        output_root=output_path.parent,
        resume_output_dir=output_path,
    )
    report = service.run_evaluation_job(job, progress_callback=lambda message: print(f"[eval_v2] {message}"))
    print(f"[E2] job={report.job_name} generated_at={report.generated_at}")
    print(f"[E2] output_dir={report.artifacts.output_dir}")
    for summary in report.summaries:
        print(
            "[E2] summary scenario={scenario} candidate={candidate} success_rate={success_rate} score_mean={score_mean} trials={trials}".format(
                scenario=summary.scenario_name,
                candidate=summary.candidate_name,
                success_rate=summary.success_rate,
                score_mean=summary.score_mean,
                trials=summary.trial_count,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
