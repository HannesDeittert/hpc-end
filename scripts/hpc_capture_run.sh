#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <run_dir> <job_id|NA> <sbatch_file> [description]" >&2
  exit 1
fi

run_dir="$1"
job_id="$2"
sbatch_file="$3"
description="${4:-}"

if [[ ! -d "$run_dir" ]]; then
  echo "run_dir not found: $run_dir" >&2
  exit 1
fi
if [[ ! -f "$sbatch_file" ]]; then
  echo "sbatch_file not found: $sbatch_file" >&2
  exit 1
fi

mkdir -p "$run_dir"
cp -f "$sbatch_file" "$run_dir/job.sbatch"

export RUN_DIR="$run_dir"
export JOB_ID="$job_id"
export DESC="$description"

sacct_out="$run_dir/job.sacct.txt"
if [[ "$job_id" == "NA" || "$job_id" == "-" || "$job_id" == "none" ]]; then
  echo "job_id not provided" > "$sacct_out"
else
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" -X \
      --format=JobID,JobName,Partition,State,Elapsed,Start,End,NodeList,ExitCode \
      --parsable2 > "$sacct_out" || true
  else
    echo "sacct not available" > "$sacct_out"
  fi
fi

python3 - <<'PY'
import json
import os
from datetime import datetime

run_dir = os.environ["RUN_DIR"]
job_id = os.environ["JOB_ID"]
desc = os.environ.get("DESC", "")
sbatch = os.path.join(run_dir, "job.sbatch")
sacct_path = os.path.join(run_dir, "job.sacct.txt")

job = {}
if os.path.exists(sacct_path):
    try:
        with open(sacct_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) >= 2 and "|" in lines[0]:
            header = lines[0].split("|")
            values = lines[1].split("|")
            job = dict(zip(header, values))
    except Exception:
        job = {}

payload = {
    "run_name": os.path.basename(run_dir),
    "run_dir": run_dir,
    "job_id": job_id,
    "job": job,
    "sbatch_file": "job.sbatch",
    "description": desc,
    "captured_at": datetime.utcnow().isoformat() + "Z",
    "user": os.environ.get("USER", ""),
    "host": os.uname().nodename,
}

with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"[ok] wrote {os.path.join(run_dir, 'run.json')}")
PY

echo "[ok] copied job.sbatch and wrote job.sacct.txt"
