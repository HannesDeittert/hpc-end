#!/usr/bin/env bash
set -euo pipefail

# Start paper-style SAC training for a stored wire (data/<model>/wires/<wire>/tool.py).
#
# Example:
#   conda activate master-project
#   source scripts/sofa_env.sh
#   bash scripts/train_paper.sh --tool TestModel_StandardJ035/StandardJ035_PTFE --name paper_standardj --device cuda --workers 16
#
# This script runs training via nohup by default and writes:
# - pid file: results/paper_runs/<name>.pid
# - nohup log: results/paper_runs/nohup_<name>.log

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR_DEFAULT="$ROOT/results/paper_runs"

tool=""
name="paper_run"
device="cuda"
workers="16"
resume_from=""
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
foreground="0"
out_dir="$OUT_DIR_DEFAULT"
tensorboard="0"
tb_logdir=""

usage() {
  cat <<EOF
Usage: bash scripts/train_paper.sh --tool <model/wire> [options]

Required:
  --tool <ref>                 Wire ref, e.g. TestModel_StandardJ035/StandardJ035_PTFE

Options:
  --name <name>                Run name (default: $name)
  --device <cpu|cuda|cuda:0>   Trainer device (default: $device)
  --workers <n>                Worker processes (default: $workers)
  --resume-from <path>         Path to .everl checkpoint (optional)
  --out <dir>                  Results root (default: $out_dir)
  --tensorboard                Enable TensorBoard logging (loss/reward/quality)
  --tb-logdir <dir>            Optional TensorBoard log dir (default: <run>/tb)
  --foreground                 Run in foreground (no nohup)
  --cuda-visible <ids>         Sets CUDA_VISIBLE_DEVICES (default: $cuda_visible_devices)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool) tool="${2:-}"; shift 2 ;;
    --name) name="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --workers) workers="${2:-}"; shift 2 ;;
    --resume-from) resume_from="${2:-}"; shift 2 ;;
    --out) out_dir="${2:-}"; shift 2 ;;
    --tensorboard) tensorboard="1"; shift 1 ;;
    --tb-logdir) tb_logdir="${2:-}"; shift 2 ;;
    --cuda-visible) cuda_visible_devices="${2:-}"; shift 2 ;;
    --foreground) foreground="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$tool" ]]; then
  echo "error: --tool is required" >&2
  usage
  exit 2
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: conda env not active (CONDA_PREFIX missing). Run: conda activate <env>" >&2
  exit 2
fi

source "$ROOT/scripts/sofa_env.sh" >/dev/null

if [[ -n "$resume_from" && ! -f "$resume_from" ]]; then
  echo "error: --resume-from not found: $resume_from" >&2
  exit 2
fi

mkdir -p "$out_dir"

export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

STEVE_TRAIN="${CONDA_PREFIX:-}/bin/steve-train"
if [[ -z "$STEVE_TRAIN" || ! -x "$STEVE_TRAIN" ]]; then
  echo "error: steve-train not found at $STEVE_TRAIN (did you run: pip install -e . ?)" >&2
  exit 2
fi

cmd=(
  "$STEVE_TRAIN"
  --tool "$tool"
  -d "$device"
  -nw "$workers"
  -n "$name"
  --stdout
  --out "$out_dir"
  --replay-device cpu
  --log-interval-s 600
  --eval-every 250000
  --eval-episodes 1
)
if [[ -n "$resume_from" ]]; then
  cmd+=(--resume-from "$resume_from")
fi
if [[ "$tensorboard" == "1" ]]; then
  cmd+=(--tensorboard)
  if [[ -n "$tb_logdir" ]]; then
    cmd+=(--tb-logdir "$tb_logdir")
  fi
fi

pid_file="$out_dir/${name}.pid"
nohup_log="$out_dir/nohup_${name}.log"

echo "[train_paper] tool=$tool name=$name device=$device workers=$workers"
echo "[train_paper] out=$out_dir cuda_visible=$CUDA_VISIBLE_DEVICES"
echo "[train_paper] cmd: ${cmd[*]}"

if [[ "$foreground" == "1" ]]; then
  exec "${cmd[@]}"
fi

nohup "${cmd[@]}" >"$nohup_log" 2>&1 & echo $! >"$pid_file"
sleep 1
echo "[train_paper] started pid=$(cat "$pid_file")"
echo "[train_paper] tail: $nohup_log"
tail -n 20 "$nohup_log" || true
