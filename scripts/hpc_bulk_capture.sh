#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_map.tsv>" >&2
  echo "Format: run_dir<TAB>job_id<TAB>sbatch_file<TAB>description" >&2
  exit 1
fi

map_file="$1"
if [[ ! -f "$map_file" ]]; then
  echo "map file not found: $map_file" >&2
  exit 1
fi

while IFS=$'\t' read -r run_dir job_id sbatch_file description; do
  [[ -z "${run_dir:-}" ]] && continue
  [[ "${run_dir:0:1}" == "#" ]] && continue
  job_id="${job_id:-NA}"
  sbatch_file="${sbatch_file:-}"
  description="${description:-}"
  if [[ -z "$sbatch_file" ]]; then
    echo "skip (missing sbatch file): $run_dir" >&2
    continue
  fi
  echo "[info] capture $run_dir (job_id=$job_id)"
  bash "$(dirname "$0")/hpc_capture_run.sh" "$run_dir" "$job_id" "$sbatch_file" "$description"
done < "$map_file"
