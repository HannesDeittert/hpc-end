from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORCE_AWARE_ROOT = PROJECT_ROOT / "data" / "archive" / "agents" / "force_aware"
WIRE_REGISTRY_ROOT = PROJECT_ROOT / "data" / "wire_registry"
DEFAULT_QUICKCHECK_SAMPLE_JSON = PROJECT_ROOT / "results" / "experimental_prep" / "sample_12_e1.json"

BASELINE_STANDARD_J_AGENT_JSONS = (
    WIRE_REGISTRY_ROOT
    / "steve_default"
    / "wire_versions"
    / "standard_j"
    / "agents"
    / "archvar_steve_standard_j_best"
    / "agent.json",
    WIRE_REGISTRY_ROOT
    / "amplatz_super_stiff"
    / "wire_versions"
    / "standard_j"
    / "agents"
    / "archvar_amplatz_standard_j_best"
    / "agent.json",
    WIRE_REGISTRY_ROOT
    / "universal_ii"
    / "wire_versions"
    / "standard_j"
    / "agents"
    / "archvar_universalii_standard_j_best"
    / "agent.json",
)

ALPHA_SUFFIX_TO_VALUE = {
    "a001": 0.01,
    "a005": 0.05,
    "a05": 0.5,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _maybe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_mean(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    header_line = text.splitlines()[0]
    delimiter = ";" if header_line.count(";") > header_line.count(",") else ","
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = []
        for row in reader:
            cleaned = {
                str(key).strip(): (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
                if key is not None
            }
            rows.append(cleaned)
    return rows


def _parse_main_log_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "latest_eval_quality": None,
        "latest_eval_reward": None,
        "latest_exploration_steps": None,
        "best_eval_quality_log": None,
        "best_eval_reward_log": None,
        "eval_count_log": 0,
    }
    quality_pattern = re.compile(
        r"Quality:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r",\s*Reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r",\s*Exploration steps:\s*(\d+)"
    )
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = quality_pattern.search(line)
        if not match:
            continue
        quality = float(match.group(1))
        reward = float(match.group(2))
        exploration_steps = int(match.group(3))
        metrics["latest_eval_quality"] = quality
        metrics["latest_eval_reward"] = reward
        metrics["latest_exploration_steps"] = exploration_steps
        metrics["best_eval_quality_log"] = (
            quality
            if metrics["best_eval_quality_log"] is None
            else max(metrics["best_eval_quality_log"], quality)
        )
        metrics["best_eval_reward_log"] = (
            reward
            if metrics["best_eval_reward_log"] is None
            else max(metrics["best_eval_reward_log"], reward)
        )
        metrics["eval_count_log"] += 1
    return metrics


def _parse_training_summary_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    if not rows:
        return {
            "summary_rows": 0,
            "summary_latest_quality": None,
            "summary_latest_reward": None,
            "summary_latest_steps_explore": None,
            "summary_best_quality": None,
            "summary_best_explore_steps": None,
        }

    latest = rows[-1]
    best_quality = None
    best_explore_steps = None
    for row in rows:
        quality = _maybe_float(row.get("quality"))
        if quality is None:
            continue
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_explore_steps = _maybe_int(row.get("steps_explore"))

    return {
        "summary_rows": len(rows),
        "summary_latest_quality": _maybe_float(latest.get("quality")),
        "summary_latest_reward": _maybe_float(latest.get("reward")),
        "summary_latest_steps_explore": _maybe_int(latest.get("steps_explore")),
        "summary_best_quality": _maybe_float(latest.get("best_quality")) or best_quality,
        "summary_best_explore_steps": _maybe_int(latest.get("best_explore_steps")) or best_explore_steps,
    }


def _parse_latest_reward_eval(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {
            "latest_reward_eval_file": None,
            "eval_episode_count": 0,
            "eval_target_success_rate": None,
            "eval_mean_total_reward": None,
            "eval_mean_path_delta": None,
            "eval_mean_force_penalty": None,
            "eval_nonzero_force_penalty_frac": None,
            "eval_mean_wire_force_normal_max_N": None,
            "eval_max_wire_force_normal_max_N": None,
            "eval_force_unavailable_frac": None,
        }

    rows = _read_csv_rows(path)
    if not rows:
        return {
            "latest_reward_eval_file": Path(path).name,
            "eval_episode_count": 0,
            "eval_target_success_rate": None,
            "eval_mean_total_reward": None,
            "eval_mean_path_delta": None,
            "eval_mean_force_penalty": None,
            "eval_nonzero_force_penalty_frac": None,
            "eval_mean_wire_force_normal_max_N": None,
            "eval_max_wire_force_normal_max_N": None,
            "eval_force_unavailable_frac": None,
        }

    target_values = [_maybe_float(row.get("target")) for row in rows]
    total_values = [_maybe_float(row.get("total")) for row in rows]
    path_delta_values = [_maybe_float(row.get("path_delta")) for row in rows]
    force_values = [_maybe_float(row.get("force")) for row in rows]
    wire_force_max_values = [_maybe_float(row.get("wire_force_normal_trial_max_N")) for row in rows]
    unavailable_count = sum(
        1
        for row in rows
        if str(row.get("force_available_for_score", "")).strip() in {"0", "False", "false"}
    )
    nonzero_force_count = sum(
        1 for value in force_values if value is not None and abs(value) > 1e-12
    )

    numeric_wire_force_max = [value for value in wire_force_max_values if value is not None]
    return {
        "latest_reward_eval_file": Path(path).name,
        "eval_episode_count": len(rows),
        "eval_target_success_rate": _safe_mean(target_values),
        "eval_mean_total_reward": _safe_mean(total_values),
        "eval_mean_path_delta": _safe_mean(path_delta_values),
        "eval_mean_force_penalty": _safe_mean(force_values),
        "eval_nonzero_force_penalty_frac": nonzero_force_count / len(rows) if rows else None,
        "eval_mean_wire_force_normal_max_N": _safe_mean(wire_force_max_values),
        "eval_max_wire_force_normal_max_N": max(numeric_wire_force_max) if numeric_wire_force_max else None,
        "eval_force_unavailable_frac": unavailable_count / len(rows) if rows else None,
    }


def _latest_reward_eval_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("reward_eval_*.csv"))
    return candidates[-1] if candidates else None


def _best_available_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    best = checkpoint_dir / "best_checkpoint.everl"
    if best.exists():
        return best
    candidates = sorted(
        path
        for path in checkpoint_dir.glob("checkpoint*.everl")
        if path.name != "latest_replay_buffer.everl"
    )
    return candidates[-1] if candidates else None


def parse_force_aware_run_name(name: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"(?P<wire>.+)_standard_j_relu_(?P<alpha_suffix>a001|a005|a05)",
        name,
    )
    if not match:
        raise ValueError(f"Unrecognized force-aware archive folder name: {name}")
    alpha_suffix = match.group("alpha_suffix")
    return {
        "wire_name": match.group("wire"),
        "wire_version": "standard_j",
        "force_alpha_suffix": alpha_suffix,
        "force_alpha": ALPHA_SUFFIX_TO_VALUE[alpha_suffix],
    }


def load_force_aware_runs(root: Path = FORCE_AWARE_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        parsed = parse_force_aware_run_name(run_dir.name)
        top_level_csvs = sorted(run_dir.glob("*.csv"))
        summary_csv = next(
            (path for path in top_level_csvs if path.name.startswith("2026-") or path.name.startswith("2027-")),
            None,
        )
        row = {
            "comparison_group": "force_aware",
            "archive_label": run_dir.name,
            "archive_run_dir": str(run_dir),
            "summary_csv": str(summary_csv) if summary_csv else None,
            "checkpoint": str(_best_available_checkpoint(run_dir)) if _best_available_checkpoint(run_dir) else None,
            **parsed,
        }
        row.update(_parse_main_log_metrics(run_dir / "main.log"))
        if summary_csv is not None:
            row.update(_parse_training_summary_csv(summary_csv))
        else:
            row.update(_parse_training_summary_csv(Path("/dev/null")))
        row.update(_parse_latest_reward_eval(_latest_reward_eval_csv(run_dir)))
        rows.append(row)
    return rows


def load_baseline_standard_j_agents(
    agent_json_paths: tuple[Path, ...] = BASELINE_STANDARD_J_AGENT_JSONS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_json_path in agent_json_paths:
        payload = _load_json(agent_json_path)
        source_checkpoint = Path(str(payload["source_checkpoint"]))
        run_dir = source_checkpoint.parent.parent
        summary_csv = next((path for path in sorted(run_dir.glob("*.csv")) if path.is_file()), None)
        tool_ref = str(payload.get("tool", ""))
        wire_name, wire_version = tool_ref.split("/", maxsplit=1)
        row = {
            "comparison_group": "baseline",
            "archive_label": payload.get("name"),
            "archive_run_dir": str(run_dir),
            "summary_csv": str(summary_csv) if summary_csv else None,
            "wire_name": wire_name,
            "wire_version": wire_version,
            "force_alpha_suffix": "baseline",
            "force_alpha": 0.0,
            "agent_json": str(agent_json_path),
            "source_checkpoint": str(source_checkpoint),
            "checkpoint": str(payload.get("checkpoint")),
        }
        row.update(_parse_main_log_metrics(run_dir / "main.log"))
        if summary_csv is not None:
            row.update(_parse_training_summary_csv(summary_csv))
        else:
            row.update(_parse_training_summary_csv(Path("/dev/null")))
        row.update(_parse_latest_reward_eval(_latest_reward_eval_csv(run_dir)))
        rows.append(row)
    return rows


def load_force_aware_comparison_rows() -> list[dict[str, Any]]:
    rows = []
    rows.extend(load_baseline_standard_j_agents())
    rows.extend(load_force_aware_runs())
    return rows


def to_frame(rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(rows)
    sort_columns = [column for column in ("wire_name", "comparison_group", "force_alpha") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    return frame


def load_sample_record_ids(
    sample_json: Path = DEFAULT_QUICKCHECK_SAMPLE_JSON,
    *,
    limit: int | None = None,
) -> list[str]:
    payload = _load_json(sample_json)
    selected = payload.get("selected_anatomies", [])
    record_ids = [str(item["record_id"]) for item in selected]
    if limit is not None:
        return record_ids[: int(limit)]
    return record_ids


def build_quickcheck_rows(
    *,
    anatomy_ids: list[str],
    force_alpha_suffixes: tuple[str, ...] = ("a001", "a005", "a05"),
    trial_count: int = 20,
    max_episode_steps: int = 500,
    base_seed: int = 123,
    workers: int = 4,
    target_branches: tuple[str, ...] = ("lcca",),
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    output_root = (
        PROJECT_ROOT / "results" / "master_thesis" / "force_aware_quickcheck"
        if output_root is None
        else Path(output_root)
    )
    rows = load_force_aware_comparison_rows()
    baseline_by_wire = {
        row["wire_name"]: row
        for row in rows
        if row.get("comparison_group") == "baseline"
        and row.get("wire_version") == "standard_j"
    }
    force_by_wire_alpha = {
        (row["wire_name"], row["force_alpha_suffix"]): row
        for row in rows
        if row.get("comparison_group") == "force_aware"
        and row.get("wire_version") == "standard_j"
    }

    job_rows: list[dict[str, Any]] = []
    for anatomy_id in anatomy_ids:
        for wire_name, baseline in baseline_by_wire.items():
            execution_wire = f"{wire_name}/standard_j"
            for alpha_suffix in force_alpha_suffixes:
                force_row = force_by_wire_alpha.get((wire_name, alpha_suffix))
                if force_row is None:
                    continue
                for candidate_kind, candidate_row in (
                    ("baseline", baseline),
                    ("force_aware", force_row),
                ):
                    checkpoint = candidate_row.get("checkpoint")
                    if not checkpoint:
                        continue
                    label = (
                        f"{wire_name}_standard_j_{candidate_kind}"
                        if candidate_kind == "baseline"
                        else f"{wire_name}_standard_j_{alpha_suffix}"
                    )
                    job_name = (
                        f"quick_{wire_name}_standard_j_{alpha_suffix}_{candidate_kind}_{anatomy_id}"
                    )
                    out_dir = output_root / job_name
                    command = " ".join(
                        [
                            "python -m steve_recommender.eval_v2.cli run",
                            f"--job-name {job_name}",
                            f"--anatomy {anatomy_id}",
                            f"--execution-wire {execution_wire}",
                            f"--policy-checkpoint {checkpoint}",
                            f"--policy-label {label}",
                            f"--policy-trained-on-wire {execution_wire}",
                            "--target-mode branch_end",
                            f"--target-branches {','.join(target_branches)}",
                            f"--trial-count {int(trial_count)}",
                            f"--base-seed {int(base_seed)}",
                            "--policy-mode deterministic",
                            "--stochastic-env-mode random_start",
                            "--policy-device cpu",
                            f"--max-episode-steps {int(max_episode_steps)}",
                            f"--workers {int(workers)}",
                            f"--output-root {output_root}",
                        ]
                    )
                    job_rows.append(
                        {
                            "anatomy_id": anatomy_id,
                            "wire_name": wire_name,
                            "wire_version": "standard_j",
                            "force_alpha_suffix": alpha_suffix,
                            "candidate_kind": candidate_kind,
                            "execution_wire": execution_wire,
                            "checkpoint": checkpoint,
                            "policy_label": label,
                            "job_name": job_name,
                            "output_dir": str(out_dir),
                            "command": command,
                        }
                    )
    return job_rows
