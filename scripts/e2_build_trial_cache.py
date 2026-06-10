#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUESTED_TRIAL_COLUMNS = [
    "trial_index",
    "candidate_name",
    "execution_wire",
    "trained_on_wire",
    "end_reason",
    "env_seed",
    "policy_seed",
    "force_available_for_score",
    "force_within_safety_threshold",
    "success",
    "steps_total",
    "tip_force_normal_trial_max_N",
    "tip_force_normal_trial_mean_N",
    "wire_force_normal_trial_max_N",
    "wire_force_normal_trial_mean_N",
]

JOB_METADATA_COLUMNS = [
    "training_family",
    "array_index",
    "source_e2_array_index",
    "anatomy_id",
    "anatomy_index",
    "target_index",
    "branch",
    "target_position",
    "target_seed",
    "seed_base",
    "config_id",
    "friction",
    "max_episode_steps",
]

DERIVED_TRIAL_COLUMNS = [
    "force_alpha",
]


def local_run_dir_from_remote(output_dir_remote: str, result_root: Path) -> Path:
    text = str(output_dir_remote)
    markers = [
        f"results/master_thesis/{result_root.name}/",
        "results/master_thesis/e2_tinyfat/",
    ]
    for marker in markers:
        if marker in text:
            return result_root / text.split(marker, maxsplit=1)[1]
    return Path(text)


def _manifest_path(result_root: Path) -> Path:
    metadata_root = result_root / "metadata"
    work_manifest = metadata_root / "job_manifest_work.json"
    force_manifest = metadata_root / "job_manifest_force_aware.json"
    if work_manifest.exists():
        return work_manifest
    if force_manifest.exists():
        return force_manifest
    raise FileNotFoundError(f"No supported E2 manifest found under {metadata_root}")


def build_manifest_df(result_root: Path) -> pd.DataFrame:
    manifest_path = _manifest_path(result_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_family = "force_aware" if manifest_path.name == "job_manifest_force_aware.json" else "non_force_aware"
    rows = []
    for idx, job in enumerate(payload["jobs"]):
        target = job.get("target_spec", {})
        config = job.get("config_spec", {})
        rows.append(
            {
                "training_family": training_family,
                "array_index": idx,
                "source_e2_array_index": job.get("source_e2_array_index", idx),
                "anatomy_id": job.get("anatomy_id"),
                "anatomy_index": job.get("anatomy_index"),
                "target_index": job.get("target_index"),
                "branch": target.get("branch"),
                "target_position": target.get("target_position"),
                "target_seed": target.get("target_seed"),
                "seed_base": job.get("seed_base"),
                "config_id": job.get("config_id"),
                "friction": job.get("friction"),
                "max_episode_steps": config.get("max_episode_steps"),
                "output_dir_remote": job.get("output_dir"),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_metadata(result_root: Path) -> pd.DataFrame:
    path = result_root / "metadata" / "force_aware_candidates.json"
    if not path.exists():
        return pd.DataFrame(columns=["candidate_name", "force_alpha"])
    rows = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        rows.append(
            {
                "candidate_name": item.get("name"),
                "force_alpha": item.get("alpha"),
                "trained_on_wire": item.get("tool_ref"),
                "force_aware_model": item.get("model"),
                "force_aware_wire": item.get("wire"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one cached E2 trial dataframe from per-run trials.csv files.")
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/master_thesis/e2_tinyfat_finished_readable_20260510"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--format", choices=("pickle", "parquet"), default="pickle")
    args = ap.parse_args()

    result_root = args.root
    out = args.out or result_root / "analysis_cache" / f"trial_df_requested_columns.{ 'pkl.gz' if args.format == 'pickle' else 'parquet' }"
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = build_manifest_df(result_root)
    candidate_metadata = build_candidate_metadata(result_root)
    alpha_by_candidate = {}
    if not candidate_metadata.empty:
        alpha_by_candidate = candidate_metadata.set_index("candidate_name")["force_alpha"].to_dict()

    frames = []
    missing_csv = 0
    for _, job in manifest_df.iterrows():
        run_dir = local_run_dir_from_remote(job["output_dir_remote"], result_root)
        csv_path = run_dir / "trials.csv"
        if not csv_path.exists():
            missing_csv += 1
            continue
        cols = [c for c in REQUESTED_TRIAL_COLUMNS if c]
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
        for col in JOB_METADATA_COLUMNS:
            df[col] = job[col]
        df["force_alpha"] = df["candidate_name"].map(alpha_by_candidate) if alpha_by_candidate else pd.NA
        frames.append(df)
        if len(frames) == 1 or len(frames) % 25 == 0:
            print(f"loaded_jobs={len(frames)} rows={sum(len(x) for x in frames)} latest={csv_path}")

    if not frames:
        raise RuntimeError(f"No trials.csv files found under {result_root}")

    trial_df = pd.concat(frames, ignore_index=True)

    # Stable column order: metadata first, then requested trial columns.
    trial_df = trial_df[
        JOB_METADATA_COLUMNS
        + [c for c in DERIVED_TRIAL_COLUMNS if c in trial_df.columns]
        + [c for c in REQUESTED_TRIAL_COLUMNS if c in trial_df.columns]
    ]

    if args.format == "parquet":
        trial_df.to_parquet(out, index=False)
    else:
        trial_df.to_pickle(out, compression="gzip")

    print("wrote", out)
    print("shape", trial_df.shape)
    print("missing_csv", missing_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
