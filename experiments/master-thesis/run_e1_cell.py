from __future__ import annotations

import argparse
import json
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
    AorticArchAnatomy,
    CenterlineRandomTarget,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationScenario,
    ExecutionPlan,
    ForceTelemetrySpec,
    ForceScoringSpec,
    FluoroscopySpec,
    PolicySpec,
    ScoringSpec,
    WireRef,
)

from e1 import (  # noqa: E402
    CONFIGS,
    DEFAULT_E1_ROOT,
    DEFAULT_WORKER_COUNT,
    build_execution_plan,
    config_spec,
    load_targets_json,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one E1 job")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--anatomy-id", default=None)
    parser.add_argument("--target-spec", default=None)
    parser.add_argument("--config-id", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--step-budget", type=int, default=None)
    parser.add_argument("--wires-json", type=Path, default=DEFAULT_E1_ROOT / "metadata" / "wires.json")
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--threshold-mm", type=float, default=5.0)
    return parser.parse_args(argv)


def _load_json(path_or_json: str | Path) -> Any:
    candidate = Path(str(path_or_json))
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(str(path_or_json))


def _parse_wire_ref(text: str) -> WireRef:
    parts = tuple(part.strip() for part in str(text).split("/", maxsplit=1))
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"wire ref must look like 'model/wire', got {text!r}")
    return WireRef(model=parts[0], wire=parts[1])


def make_default_service():
    from steve_recommender.eval_v2.service import DefaultEvaluationService

    return DefaultEvaluationService()


def _row_from_manifest(manifest_path: Path, index: int) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if index < 0 or index >= len(jobs):
        raise IndexError(f"array index {index} out of range for {manifest_path}")
    return dict(jobs[index])


def _load_wires(wires_json: Path, service: DefaultEvaluationService) -> tuple[WireRef, ...]:
    if wires_json.exists():
        payload = json.loads(wires_json.read_text(encoding="utf-8"))
        wires = []
        for item in payload.get("wires", []):
            tool_ref = str(item.get("tool_ref", ""))
            if "/" not in tool_ref:
                continue
            wires.append(_parse_wire_ref(tool_ref))
        if wires:
            return tuple(wires)
    return tuple(service.list_startable_wires())


def _build_candidate(
    *,
    service: Any,
    execution_wire: WireRef,
) -> EvaluationCandidate:
    candidates = service.list_candidates(execution_wire=execution_wire, include_cross_wire=False)
    if not candidates:
        raise ValueError(f"No candidates available for {execution_wire.tool_ref}")
    return candidates[0]


def _build_scenario(
    *,
    anatomy_id: str,
    target_spec: dict[str, Any],
    service: Any,
) -> EvaluationScenario:
    anatomy = service.get_anatomy(record_id=anatomy_id)
    target = CenterlineRandomTarget(
        threshold_mm=float(target_spec.get("threshold_mm", 5.0)),
        branches=tuple(str(branch) for branch in target_spec.get("branches", ())),
        seed=int(target_spec["target_seed"]),
    )
    return EvaluationScenario(
        name=f"{anatomy_id}__target_{int(target_spec.get('target_index', 0))}",
        anatomy=anatomy,
        target=target,
        fluoroscopy=FluoroscopySpec(),
        force_telemetry=ForceTelemetrySpec(
            mode="passive",
            required=False,
            tip_threshold_mm=float(target_spec.get("threshold_mm", 5.0)),
            write_full_trace=True,
            write_diagnostics=False,
            plugin_path=None,
            units=None,
        ),
    )


def _build_execution_plan(
    *,
    config_id: int,
    seed_base: int,
    trial_count: int,
    max_episode_steps: int,
    worker_count: int,
) -> ExecutionPlan:
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


def _job_from_inputs(
    *,
    anatomy_id: str,
    target_spec: dict[str, Any],
    config_id: int,
    num_trials: int,
    seed_base: int,
    output_path: Path,
    step_budget: int,
    worker_count: int,
    policy_device: str,
    threshold_mm: float,
    wires_json: Path,
    service: Any,
) -> EvaluationJob:
    _ = policy_device, threshold_mm
    scenario = _build_scenario(anatomy_id=anatomy_id, target_spec=target_spec, service=service)
    wires = _load_wires(wires_json, service)
    candidates = tuple(_build_candidate(service=service, execution_wire=wire) for wire in wires)
    execution = _build_execution_plan(
        config_id=config_id,
        seed_base=seed_base,
        trial_count=num_trials,
        max_episode_steps=step_budget,
        worker_count=worker_count,
    )
    return EvaluationJob(
        name=Path(output_path).name,
        scenarios=(scenario,),
        candidates=candidates,
        execution=execution,
        scoring=ScoringSpec(force=ForceScoringSpec(tip_length_mm=float(threshold_mm))),
        output_root=Path(output_path).parent,
        resume_output_dir=Path(output_path),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    service = make_default_service()

    if args.manifest is not None:
        if args.array_index is None:
            raise ValueError("--array-index is required when --manifest is used")
        row = _row_from_manifest(args.manifest, args.array_index)
        anatomy_id = str(row["anatomy_id"])
        target_spec = dict(row["target_spec"])
        config_id = int(row["config_id"])
        num_trials = int(row["config_spec"]["trial_count"]) if args.num_trials is None else int(args.num_trials)
        seed_base = int(row["seed_base"]) if args.seed_base is None else int(args.seed_base)
        output_path = Path(str(row["output_dir"])) if args.output_path is None else Path(args.output_path)
        step_budget = int(args.step_budget) if args.step_budget is not None else int(row["config_spec"].get("max_episode_steps", 1000))
    else:
        if args.anatomy_id is None or args.target_spec is None or args.config_id is None or args.output_path is None or args.seed_base is None or args.step_budget is None:
            raise ValueError(
                "Either --manifest/--array-index or the direct arguments "
                "--anatomy-id --target-spec --config-id --num-trials --seed-base --output-path --step-budget must be provided"
            )
        anatomy_id = str(args.anatomy_id)
        target_spec = _load_json(args.target_spec)
        config_id = int(args.config_id)
        num_trials = int(args.num_trials or config_spec(config_id)["trial_count"])
        seed_base = int(args.seed_base)
        output_path = Path(args.output_path)
        step_budget = int(args.step_budget)

    job = _job_from_inputs(
        anatomy_id=anatomy_id,
        target_spec=target_spec,
        config_id=config_id,
        num_trials=num_trials,
        seed_base=seed_base,
        output_path=output_path,
        step_budget=step_budget,
        worker_count=int(args.worker_count),
        policy_device=str(args.policy_device),
        threshold_mm=float(args.threshold_mm),
        wires_json=Path(args.wires_json),
        service=service,
    )

    report = service.run_evaluation_job(
        job,
        progress_callback=lambda message: print(f"[eval_v2] {message}"),
    )
    print(f"[E1] job={report.job_name} generated_at={report.generated_at}")
    print(f"[E1] output_dir={report.artifacts.output_dir}")
    for summary in report.summaries:
        print(
            "[E1] summary scenario={scenario} candidate={candidate} success_rate={success_rate} score_mean={score_mean} trials={trials}".format(
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
