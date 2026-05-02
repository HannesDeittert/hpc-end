from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_JSON = PROJECT_ROOT / "results" / "experimental_prep" / "sample_12.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e0"
DEFAULT_TARGET_BRANCHES = ("bct", "lcca", "lsa")


def _parse_csv_list(text: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("List argument must not be empty")
    return values


def _parse_wire_ref(text: str) -> Tuple[str, str]:
    parts = tuple(part.strip() for part in str(text).split("/", maxsplit=1))
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"wire ref must look like 'model/wire', got {text!r}")
    return parts[0], parts[1]


def _load_sampled_anatomies(sample_json: Path) -> Tuple[str, ...]:
    payload = json.loads(Path(sample_json).read_text(encoding="utf-8"))
    selected = payload.get("selected_anatomies")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"No selected_anatomies found in {sample_json}")
    return tuple(str(item["record_id"]) for item in selected)


def _read_anatomy_description(record_id: str) -> Dict[str, object]:
    desc_path = PROJECT_ROOT / "data" / "anatomy_registry" / "anatomies" / record_id / "description.json"
    payload = json.loads(desc_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid anatomy description: {desc_path}")
    return payload


def _load_branch_point_counts(record_id: str, branch_names: Sequence[str]) -> Dict[str, int]:
    desc = _read_anatomy_description(record_id)
    centerline_rel = desc.get("centerline_bundle_path") or desc.get("centerline_npz") or "centerline.npz"
    centerline_path = (
        PROJECT_ROOT
        / "data"
        / "anatomy_registry"
        / "anatomies"
        / record_id
        / str(centerline_rel)
    )
    with np.load(centerline_path, allow_pickle=True) as data:
        counts: Dict[str, int] = {}
        for branch in branch_names:
            key = f"branch_{branch}_coords"
            if key not in data.files:
                raise KeyError(f"{record_id} does not contain branch {branch!r}")
            counts[branch] = int(np.asarray(data[key]).shape[0])
    return counts


def _choose_target(
    *,
    record_id: str,
    env_seed: int,
    target_branches: Sequence[str],
) -> Tuple[str, int]:
    available_counts = _load_branch_point_counts(record_id, target_branches)
    available = [branch for branch in target_branches if branch in available_counts]
    if not available:
        raise ValueError(f"{record_id} has none of the requested target branches: {target_branches}")
    rng = random.Random(int(env_seed))
    branch = rng.choice(available)
    index = rng.randrange(int(available_counts[branch]))
    return branch, index


def _list_candidates(
    *,
    execution_wire: str,
    include_cross_wire: bool,
) -> List[Tuple[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "steve_recommender.eval_v2.cli",
        "list-candidates",
        "--execution-wire",
        execution_wire,
    ]
    if not include_cross_wire:
        cmd.append("--no-cross-wire")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    candidates: List[Tuple[str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        fields: Dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", maxsplit=1)
            fields[key] = value
        name = fields.get("name")
        execution = fields.get("execution")
        if name and execution:
            candidates.append((name, execution))
    return candidates


def _build_cli_command(
    *,
    record_id: str,
    env_seed: int,
    branch: str,
    index: int,
    candidate_name: str,
    execution_wire: str,
    args: argparse.Namespace,
) -> List[str]:
    env_seeds = ",".join(str(env_seed) for _ in range(int(args.trial_count)))
    job_name = f"{record_id}_seed_{env_seed}_{candidate_name}"
    scenario_name = f"{record_id}_seed_{env_seed}"
    return [
        sys.executable,
        "-m",
        "steve_recommender.eval_v2.cli",
        "run",
        "--job-name",
        job_name,
        "--scenario-name",
        scenario_name,
        "--anatomy",
        record_id,
        "--execution-wire",
        execution_wire,
        "--candidate-name",
        candidate_name,
        "--target-mode",
        "branch_index",
        "--target-branch",
        branch,
        "--target-index",
        str(index),
        "--trial-count",
        str(int(args.trial_count)),
        "--base-seed",
        str(env_seed),
        "--env-seeds",
        env_seeds,
        "--policy-mode",
        "deterministic",
        "--stochastic-env-mode",
        "random_start",
        "--max-episode-steps",
        str(int(args.max_episode_steps)),
        "--workers",
        str(int(args.worker_count)),
        "--policy-device",
        str(args.policy_device),
        "--output-root",
        str(args.output_root),
        "--threshold-mm",
        str(float(args.threshold_mm)),
        "--friction",
        str(float(args.friction)),
        "--tip-length-mm",
        str(float(args.tip_length_mm)),
        "--image-frequency-hz",
        str(float(args.image_frequency_hz)),
        "--image-rot-z-deg",
        str(float(args.image_rot_z_deg)),
        "--image-rot-x-deg",
        str(float(args.image_rot_x_deg)),
    ]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the E0 master-thesis experiment as a thin eval_v2 CLI loop."
    )
    parser.add_argument(
        "--sample-json",
        type=Path,
        default=DEFAULT_SAMPLE_JSON,
        help="Path to the sampled anatomy JSON created in the notebook.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for eval_v2 result folders.",
    )
    parser.add_argument(
        "--execution-wire",
        action="append",
        required=True,
        help="Execution wire formatted as 'model/wire'. Repeat for multiple wires.",
    )
    parser.add_argument(
        "--include-cross-wire",
        action="store_true",
        help="Include cross-wire candidates when resolving candidates for each execution wire.",
    )
    parser.add_argument(
        "--target-branches",
        type=_parse_csv_list,
        default=DEFAULT_TARGET_BRANCHES,
        help="Comma-separated branch names used for the random target selection.",
    )
    parser.add_argument(
        "--trial-count",
        type=int,
        default=100,
        help="Number of trials per candidate and experiment.",
    )
    parser.add_argument(
        "--runs-per-anatomy",
        type=int,
        default=3,
        help="How many experiments to run for each anatomy.",
    )
    parser.add_argument(
        "--env-seed-start",
        type=int,
        default=123,
        help="First environmental seed used for the experiment loop.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=12,
        help="Parallel worker count for headless eval_v2 runs.",
    )
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--threshold-mm", type=float, default=5.0)
    parser.add_argument("--friction", type=float, default=0.001)
    parser.add_argument("--tip-length-mm", type=float, default=3.0)
    parser.add_argument("--image-frequency-hz", type=float, default=7.5)
    parser.add_argument("--image-rot-z-deg", type=float, default=20.0)
    parser.add_argument("--image-rot-x-deg", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sample_ids = _load_sampled_anatomies(args.sample_json)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[str, str]] = []
    for execution_wire in args.execution_wire:
        candidates.extend(
            _list_candidates(
                execution_wire=execution_wire,
                include_cross_wire=bool(args.include_cross_wire),
            )
        )
    deduped_candidates: List[Tuple[str, str]] = []
    seen = set()
    for candidate_name, execution_wire in candidates:
        key = (candidate_name, execution_wire)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append((candidate_name, execution_wire))
    candidates = deduped_candidates
    if not candidates:
        raise ValueError("No candidates resolved for the requested execution wires")

    manifest_rows: List[Dict[str, object]] = []
    env_seed = int(args.env_seed_start)
    cli_manifest_path = output_root / "commands.json"

    for record_id in sample_ids:
        for repeat_index in range(int(args.runs_per_anatomy)):
            branch, index = _choose_target(
                record_id=record_id,
                env_seed=env_seed,
                target_branches=args.target_branches,
            )
            for candidate_name, execution_wire in candidates:
                cmd = _build_cli_command(
                    record_id=record_id,
                    env_seed=env_seed,
                    branch=branch,
                    index=index,
                    candidate_name=candidate_name,
                    execution_wire=execution_wire,
                    args=args,
                )
                print("[E0] " + " ".join(cmd))
                subprocess.run(cmd, check=True)
                manifest_rows.append(
                    {
                        "record_id": record_id,
                        "env_seed": env_seed,
                        "repeat_index": repeat_index,
                        "target_branch": branch,
                        "target_index": index,
                        "candidate_name": candidate_name,
                        "execution_wire": execution_wire,
                        "job_name": f"{record_id}_seed_{env_seed}_{candidate_name}",
                    }
                )
            env_seed += 1

    manifest = {
        "sample_json": Path(args.sample_json).as_posix(),
        "output_root": output_root.as_posix(),
        "env_seed_start": int(args.env_seed_start),
        "runs_per_anatomy": int(args.runs_per_anatomy),
        "trial_count": int(args.trial_count),
        "worker_count": int(args.worker_count),
        "execution_wires": list(args.execution_wire),
        "target_branches": list(args.target_branches),
        "selected_anatomies": sample_ids,
        "commands": manifest_rows,
    }
    cli_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[E0] wrote manifest to {cli_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
