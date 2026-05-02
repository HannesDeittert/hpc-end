#!/bin/bash -l
#SBATCH --job-name=e1_work
#SBATCH --partition=work
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-143
#SBATCH --output=/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e1_smoke/logs/slurm-%A_%a.out
#SBATCH --error=/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e1_smoke/logs/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

cd /home/hannes-deittert/dev/Uni/master-project
eval "$(conda shell.bash hook)"
conda activate "${E1_CONDA_ENV:-/home/woody/iwhr/iwhr106h/conda/envs/master-project}"
export SOFA_ROOT="${SOFA_ROOT:-/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux}"
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="${STEVE_WALL_FORCE_MONITOR_PLUGIN:-$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export E1_JOB_MANIFEST="/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e1_smoke/metadata/job_manifest_work.json"

exec "$PYTHON_BIN" experiments/master-thesis/run_e1_cell.py --manifest "$E1_JOB_MANIFEST" --array-index "${SLURM_ARRAY_TASK_ID}"
