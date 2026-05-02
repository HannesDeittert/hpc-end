from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_JSON = PROJECT_ROOT / 'results' / 'experimental_prep' / 'sample_12.json'
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / 'results' / 'master_thesis' / 'e0_chunks'
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / 'results' / 'master_thesis' / 'e0_analysis'


def split_sample_json_into_chunks(
    sample_json: Path,
    *,
    chunk_size: int = 3,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Split one sample file into chunk files and return a compact overview table."""
    sample_json = Path(sample_json).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else sample_json.parent
    payload = json.loads(sample_json.read_text(encoding='utf-8'))
    selected = payload.get('selected_anatomies', [])
    if not selected:
        raise ValueError(f'No selected_anatomies found in {sample_json}')

    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for chunk_index, start in enumerate(range(0, len(selected), chunk_size)):
        chunk_items = selected[start : start + chunk_size]
        chunk_payload = dict(payload)
        chunk_payload['selected_anatomies'] = chunk_items
        chunk_path = output_dir / f'{sample_json.stem}_chunk_{chunk_index}.json'
        chunk_path.write_text(
            json.dumps(chunk_payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        rows.append(
            {
                'chunk_id': chunk_index,
                'chunk_path': str(chunk_path),
                'n_anatomies': len(chunk_items),
                'record_ids': ', '.join(item['record_id'] for item in chunk_items),
            }
        )

    return rows


def build_chunked_runner_script(
    project_root: Path,
    *,
    array_size: int = 4,
    partition: str = 'work',
    gpus: int = 4,
    cpus_per_task: int = 32,
    walltime: str = '24:00:00',
    worker_count: int = 29,
    runs_per_anatomy: int = 3,
    trial_count: int = 100,
    max_episode_steps: int = 1000,
    base_seed_start: int = 123,
    target_seed_start: int = 9000,
    base_seed_step: int = 200,
) -> str:
    project_root = Path(project_root).resolve()
    return f"""#!/bin/bash -l
#SBATCH --job-name=e0_chunks
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={walltime}
#SBATCH --array=0-{array_size - 1}
#SBATCH --output=results/master_thesis/e0_chunks/slurm-%A_%a.out
#SBATCH --error=results/master_thesis/e0_chunks/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

module purge
module load python
module load cmake
module load boost

eval "$(conda shell.bash hook)"
conda activate /home/woody/iwhr/iwhr106h/conda/envs/master-project

cd {project_root}
export SOFA_ROOT=/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so"

CHUNK_ID="${{SLURM_ARRAY_TASK_ID}}"
CHUNK_FILE="results/experimental_prep/sample_12_chunk_${{CHUNK_ID}}.json"
OUTPUT_ROOT="results/master_thesis/e0_chunks/chunk_${{CHUNK_ID}}"

RUNS_PER_ANATOMY={runs_per_anatomy}
TRIAL_COUNT={trial_count}
MAX_EPISODE_STEPS={max_episode_steps}
WORKER_COUNT={worker_count}
BASE_SEED_START=$(({base_seed_start} + {base_seed_step} * CHUNK_ID * 3 * RUNS_PER_ANATOMY))
TARGET_SEED_START=$(({target_seed_start} + CHUNK_ID * 3 * RUNS_PER_ANATOMY))

mkdir -p "$OUTPUT_ROOT"

SAMPLE_JSON="$CHUNK_FILE" OUTPUT_ROOT="$OUTPUT_ROOT" RUNS_PER_ANATOMY="$RUNS_PER_ANATOMY" TRIAL_COUNT="$TRIAL_COUNT" MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" WORKER_COUNT="$WORKER_COUNT" BASE_SEED_START="$BASE_SEED_START" TARGET_SEED_START="$TARGET_SEED_START" bash experiments/master-thesis/run_e0_experiment.sh
"""


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _decode_h5_column(array: Any) -> list[Any]:
    import numpy as np

    values = np.asarray(array)
    if values.dtype.kind in {'S', 'O'}:
        decoded: list[Any] = []
        for value in values:
            if isinstance(value, (bytes, bytearray)):
                decoded.append(value.decode('utf-8'))
            elif value is None:
                decoded.append(None)
            else:
                decoded.append(value)
        return decoded
    return values.tolist()


def _maybe_int(value: Any):
    if value in ('', None):
        return None
    try:
        return int(value)
    except Exception:
        return value


def _maybe_float(value: Any):
    if value in ('', None):
        return None
    try:
        return float(value)
    except Exception:
        return value


def _maybe_bool(value: Any):
    if isinstance(value, bool):
        return value
    if value in ('1', 1, 'true', 'True', 'TRUE'):
        return True
    if value in ('0', 0, 'false', 'False', 'FALSE'):
        return False
    return bool(value)


def load_trials_h5(path: Path) -> list[dict[str, Any]]:
    import h5py

    path = Path(path)
    with h5py.File(path, 'r') as handle:
        group = handle['trials']
        columns = {name: _decode_h5_column(group[name][...]) for name in group.keys()}

    n_rows = len(next(iter(columns.values()))) if columns else 0
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        row = {name: values[idx] for name, values in columns.items()}
        for key in ['success', 'valid_for_ranking', 'force_within_safety_threshold', 'force_available_for_score']:
            if key in row:
                row[key] = _maybe_bool(row[key])
        for key in ['trial_index', 'env_seed', 'max_episode_steps', 'steps_total']:
            if key in row:
                row[key] = _maybe_int(row[key])
        for key in [
            'policy_seed',
            'episode_reward',
            'sim_time_s',
            'wall_time_s',
            'tip_speed_max_mm_s',
            'tip_speed_mean_mm_s',
            'tip_total_distance_mm',
            'steps_to_success',
            'wire_force_magnitude_instant_N',
            'wire_force_magnitude_trial_max_N',
            'wire_force_magnitude_trial_mean_N',
            'wire_force_normal_instant_N',
            'wire_force_normal_trial_max_N',
            'wire_force_normal_trial_mean_N',
            'tip_force_magnitude_instant_N',
            'tip_force_magnitude_trial_max_N',
            'tip_force_magnitude_trial_mean_N',
            'tip_force_normal_instant_N',
            'tip_force_normal_trial_max_N',
            'tip_force_normal_trial_mean_N',
            'tip_length_mm',
            'tip_acc_p95',
            'tip_acc_max',
            'tip_jerk_p95',
            'tip_jerk_max',
            'score_success',
            'score_efficiency',
            'score_safety',
            'score_smoothness',
            'score_total',
        ]:
            if key in row:
                row[key] = _maybe_float(row[key])
        rows.append(row)
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def load_e0_results(result_root: Path):
    result_root = Path(result_root).resolve()
    manifest_paths = sorted(result_root.rglob('manifest.json'))
    if not manifest_paths:
        raise FileNotFoundError(
            f'No manifest.json files found under {result_root}. Sync the chunk results there or set E0_RESULTS_ROOT.'
        )

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        run_dir = manifest_path.parent
        artifact_paths = payload.get('artifact_paths', {})

        candidate_csv = _resolve_path(
            run_dir,
            artifact_paths.get('candidate_summaries_csv', run_dir / 'candidate_summaries.csv'),
        )
        trials_h5 = _resolve_path(run_dir, artifact_paths.get('trials_h5', run_dir / 'trials.h5'))

        run_meta = {
            'job_name': payload.get('job_name', run_dir.name),
            'run_dir': str(run_dir),
            'manifest_path': str(manifest_path),
            'chunk_dir': run_dir.parent.name if run_dir.parent.name.startswith('chunk_') else '',
            'generated_time': payload.get('generated_time'),
            'n_anatomies': len(payload.get('anatomy_metadata', [])),
            'n_trials_total': payload.get('counts', {}).get('n_trials_total'),
        }
        manifest_rows.append(run_meta)

        if candidate_csv.exists():
            for row in _load_csv_rows(candidate_csv):
                row.update(run_meta)
                row['source_file'] = str(candidate_csv)
                summary_rows.append(row)

        if trials_h5.exists():
            for row in load_trials_h5(trials_h5):
                row.update(run_meta)
                row['source_file'] = str(trials_h5)
                trial_rows.append(row)

    return manifest_rows, summary_rows, trial_rows


def _safe_mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def _safe_std(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.pstdev(values) if len(values) > 1 else None


def _safe_percentile(values: Iterable[float], q: float) -> float | None:
    import numpy as np

    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})


def analyze_e0_results(
    trial_rows: list[dict[str, Any]],
    *,
    analysis_root: Path,
    current_max_default: int = 1000,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    analysis_root = Path(analysis_root).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    if not trial_rows:
        raise ValueError('trial_rows is empty; there is nothing to analyze.')

    for row in trial_rows:
        row['steps_total'] = _maybe_float(row.get('steps_total'))
        row['steps_to_success'] = _maybe_float(row.get('steps_to_success'))
        row['score_total'] = _maybe_float(row.get('score_total'))
        row['success'] = _maybe_bool(row.get('success'))
        row['max_episode_steps'] = _maybe_int(row.get('max_episode_steps'))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        groups.setdefault(str(row.get('execution_wire', '')), []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    for wire, rows in groups.items():
        score_values = [row.get('score_total') for row in rows]
        success_values = [row.get('success') for row in rows]
        steps_total_values = [row.get('steps_total') for row in rows]
        steps_to_success_values = [row.get('steps_to_success') for row in rows if row.get('success') and row.get('steps_to_success') is not None]

        success_rate = _safe_mean([1.0 if bool(v) else 0.0 for v in success_values])
        score_mean = _safe_mean(score_values)
        score_std = _safe_std(score_values)
        steps_total_mean = _safe_mean(steps_total_values)
        steps_total_p95 = _safe_percentile(steps_total_values, 0.95)
        steps_to_success_mean = _safe_mean(steps_to_success_values)
        steps_to_success_p95 = _safe_percentile(steps_to_success_values, 0.95)

        candidate_rows.append(
            {
                'execution_wire': wire,
                'n_trials': len(rows),
                'success_rate': success_rate,
                'score_mean': score_mean,
                'score_std': score_std,
                'steps_total_mean': steps_total_mean,
                'steps_total_p95': steps_total_p95,
                'steps_to_success_mean': steps_to_success_mean,
                'steps_to_success_p95': steps_to_success_p95,
            }
        )

    candidate_rows.sort(
        key=lambda r: (
            r['score_mean'] is None,
            -(r['score_mean'] if r['score_mean'] is not None else float('-inf')),
            -(r['success_rate'] if r['success_rate'] is not None else float('-inf')),
        )
    )

    successful_steps = [float(row['steps_to_success']) for row in trial_rows if row.get('success') and row.get('steps_to_success') is not None]
    all_steps = [float(row['steps_total']) for row in trial_rows if row.get('steps_total') is not None]
    current_max = max((int(row['max_episode_steps']) for row in trial_rows if row.get('max_episode_steps') is not None), default=current_max_default)

    if successful_steps:
        percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_rows = [
            {'quantile': int(q * 100), 'steps_to_success': _safe_percentile(successful_steps, q)}
            for q in percentiles
        ]
        p95 = next((row['steps_to_success'] for row in percentile_rows if row['quantile'] == 95), None)
        recommended_cutoff = int(math.ceil(float(p95) / 25.0) * 25) if p95 is not None else None
    else:
        percentile_rows = []
        recommended_cutoff = None

    _write_rows_csv(analysis_root / 'e0_candidate_aggregate.csv', candidate_rows, [
        'execution_wire', 'n_trials', 'success_rate', 'score_mean', 'score_std', 'steps_total_mean', 'steps_total_p95', 'steps_to_success_mean', 'steps_to_success_p95',
    ])
    _write_rows_csv(analysis_root / 'e0_trials_flat.csv', trial_rows, sorted({key for row in trial_rows for key in row.keys()}))

    if candidate_rows:
        labels = [row['execution_wire'] for row in candidate_rows]
        success_rates = [row['success_rate'] if row['success_rate'] is not None else float('nan') for row in candidate_rows]
        steps_success_mean = [row['steps_to_success_mean'] if row['steps_to_success_mean'] is not None else float('nan') for row in candidate_rows]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
        y = np.arange(len(labels))
        axes[0].barh(y, success_rates, color='#2a6fdb')
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel('success rate')
        axes[0].set_title('E0 candidate success rate')
        axes[0].invert_yaxis()

        axes[1].barh(y, steps_success_mean, color='#e07a2d')
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([])
        axes[1].set_xlabel('mean steps to success')
        axes[1].set_title('Mean steps needed for successful rollouts')
        axes[1].invert_yaxis()

        fig.savefig(analysis_root / 'e0_candidate_summary.png', dpi=200, bbox_inches='tight')
        fig.savefig(analysis_root / 'e0_candidate_summary.pdf', bbox_inches='tight')
        plt.close(fig)

    step_axis_max = int(max(current_max, max(all_steps, default=0), max(successful_steps, default=0)))
    if successful_steps:
        cutoffs = list(range(25, step_axis_max + 25, 25))
        completion_curve = [
            {'step_cutoff': cutoff, 'share_of_successes_finished': sum(1 for step in successful_steps if step <= cutoff) / len(successful_steps)}
            for cutoff in cutoffs
        ]

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        ax.plot(
            [row['step_cutoff'] for row in completion_curve],
            [row['share_of_successes_finished'] for row in completion_curve],
            color='#0f766e',
            linewidth=2.5,
        )
        ax.axvline(current_max, color='#6b7280', linestyle='--', linewidth=1.8, label=f'current max = {current_max}')
        if recommended_cutoff is not None:
            ax.axvline(recommended_cutoff, color='#dc2626', linestyle='--', linewidth=1.8, label=f'95th percentile = {recommended_cutoff}')
        ax.set_xlim(0, step_axis_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('step budget cutoff')
        ax.set_ylabel('share of successful trials completed')
        ax.set_title('How much of the success distribution is captured by a step budget?')
        ax.legend(loc='lower right')
        fig.savefig(analysis_root / 'e0_step_budget_curve.png', dpi=200, bbox_inches='tight')
        fig.savefig(analysis_root / 'e0_step_budget_curve.pdf', bbox_inches='tight')
        plt.close(fig)

        def _ecdf(values: list[float]):
            data = np.sort(np.asarray(values, dtype=float))
            y = np.arange(1, len(data) + 1) / len(data)
            return data, y

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        all_x, all_y = _ecdf(all_steps)
        succ_x, succ_y = _ecdf(successful_steps)
        ax.step(all_x, all_y, where='post', color='#2563eb', linewidth=2, label='all trials: steps_total')
        ax.step(succ_x, succ_y, where='post', color='#16a34a', linewidth=2, label='successful trials: steps_to_success')
        ax.axvline(current_max, color='#6b7280', linestyle='--', linewidth=1.5)
        if recommended_cutoff is not None:
            ax.axvline(recommended_cutoff, color='#dc2626', linestyle='--', linewidth=1.5)
        ax.set_xlabel('steps')
        ax.set_ylabel('cumulative share')
        ax.set_title('Episode-length distribution')
        ax.legend(loc='lower right')
        fig.savefig(analysis_root / 'e0_episode_length_ecdf.png', dpi=200, bbox_inches='tight')
        fig.savefig(analysis_root / 'e0_episode_length_ecdf.pdf', bbox_inches='tight')
        plt.close(fig)

    return {
        'candidate_rows': candidate_rows,
        'successful_steps': successful_steps,
        'all_steps': all_steps,
        'percentiles_rows': percentile_rows,
        'recommended_cutoff': recommended_cutoff,
        'current_max': current_max,
        'analysis_root': analysis_root,
        'step_axis_max': step_axis_max,
    }
