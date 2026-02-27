#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <user@host> <remote_results_root> [local_dir]" >&2
  echo "Example: $0 iwhr106h@tinyx.nhr.fau.de /home/woody/iwhr/iwhr106h/master-project/results logs/hpc/raw" >&2
  exit 1
fi

remote="$1"
remote_root="$2"
local_dir="${3:-logs/hpc/raw}"

mkdir -p "$local_dir"

rsync -av --include "*/" --include "main.log" --include "*.csv" --include "run.json" --include "job.sbatch" --include "job.sacct.txt" --exclude "*" \
  "${remote}:${remote_root}/" "$local_dir/"

echo "[ok] synced to $local_dir"
