#!/usr/bin/env bash
set -euo pipefail

# Monitor a running training started via scripts/train_paper.sh
#
# Examples:
#   bash scripts/monitor_paper.sh --name paper_standardj
#   bash scripts/monitor_paper.sh --log results/paper_runs/nohup_paper_standardj.log

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR_DEFAULT="$ROOT/results/paper_runs"

name=""
log=""
out_dir="$OUT_DIR_DEFAULT"

usage() {
  cat <<EOF
Usage: bash scripts/monitor_paper.sh [--name <run_name>] [--log <nohup_log>] [--out <dir>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="${2:-}"; shift 2 ;;
    --log) log="${2:-}"; shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$log" ]]; then
  if [[ -z "$name" ]]; then
    echo "error: provide --name or --log" >&2
    usage
    exit 2
  fi
  log="$out_dir/nohup_${name}.log"
fi

if [[ ! -f "$log" ]]; then
  echo "error: log not found: $log" >&2
  exit 2
fi

echo "[monitor] log=$log"
echo "[monitor] tip: Ctrl+C stops tail (does not stop training)"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[monitor] GPU snapshot:"
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits || true
  echo
fi

tail -f "$log"

