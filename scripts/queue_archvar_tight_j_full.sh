#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DEFAULT="$ROOT/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94"
QUEUE_NAME_DEFAULT="queue_archvar_tight_j_full_$(date +%Y%m%d_%H%M%S)"

results="$RESULTS_DEFAULT"
queue_name="$QUEUE_NAME_DEFAULT"
device="cuda"
workers="22"

usage() {
  cat <<EOF
Usage: bash scripts/queue_archvar_tight_j_full.sh [options]

Options:
  --results <dir>       Results folder (default: $results)
  --queue-name <name>   Queue name prefix (default: $queue_name)
  --device <device>     Trainer device (default: $device)
  --workers <n>         Worker count (default: $workers)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) results="${2:-}"; shift 2 ;;
    --queue-name) queue_name="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --workers) workers="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: activate the master-thesis conda env before starting the queue" >&2
  exit 2
fi

mkdir -p "$results"
echo $$ > "$results/${queue_name}.pid"

export SOFA_ROOT="${SOFA_ROOT:-/data/H_deittert/opt/SOFA_v23.06.00_Linux}"
source "$ROOT/scripts/sofa_env.sh"
export MPLCONFIGDIR=/tmp/mplconfig
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cleanup() {
  rm -f "$results/${queue_name}.current"
}

trap cleanup EXIT

run_job() {
  local prefix="$1"
  local module="$2"
  local cls="$3"

  local run_name
  run_name="${prefix}_$(date +%Y%m%d_%H%M%S)"

  echo "[queue] starting $run_name"
  echo "$run_name" > "$results/${queue_name}.current"
  echo "$run_name" >> "$results/${queue_name}.started"

  python "$ROOT/scripts/archvar_train_tool.py" \
    -d "$device" \
    -nw "$workers" \
    -lr 0.0003218 \
    --hidden 400 400 400 \
    -en 900 \
    -el 1 \
    -n "$run_name" \
    --tool-module "$module" \
    --tool-class "$cls" \
    --results-folder "$results"

  echo "[queue] finished $run_name"
  echo "$run_name" >> "$results/${queue_name}.finished"
}

run_job \
  archvar_amplatz_tight_j_w22 \
  steve_recommender.bench.custom_tools_amplatz_tight_j_simple \
  JShapedAmplatzSuperStiffTightJSimple

run_job \
  archvar_steve_tight_j_w22 \
  steve_recommender.bench.custom_tools_steve_tight_j_simple \
  JShapedDefaultTightJSimple

run_job \
  archvar_universalii_tight_j_w22 \
  steve_recommender.bench.custom_tools_universalii_tight_j_simple \
  JShapedUniversalIITightJSimple

echo "[queue] all jobs done"
