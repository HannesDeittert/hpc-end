#!/bin/bash -l
#SBATCH --job-name=e1_tinyfat_work
#SBATCH --partition=work
#SBATCH --cpus-per-task=32
#SBATCH --hint=multithread
#SBATCH --time=24:00:00
#SBATCH --array=0-143
#SBATCH --output=/home/woody/iwhr/iwhr106h/master-project/results/master_thesis/e1_tinyfat_friction_01/logs/slurm-%A_%a.out
#SBATCH --error=/home/woody/iwhr/iwhr106h/master-project/results/master_thesis/e1_tinyfat_friction_01/logs/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

cd /home/woody/iwhr/iwhr106h/master-project
module load python
eval "$(conda shell.bash hook)"
conda activate "${E1_CONDA_ENV:-/home/woody/iwhr/iwhr106h/conda/envs/master-project}"
export SOFA_ROOT="${SOFA_ROOT:-/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux}"
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="${STEVE_WALL_FORCE_MONITOR_PLUGIN:-$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export E1_JOB_MANIFEST="/home/woody/iwhr/iwhr106h/master-project/results/master_thesis/e1_tinyfat_friction_01/metadata/job_manifest_work.json"
export E1_WORKER_COUNT="60"
export E1_FRICTION="0.1"

exec "$PYTHON_BIN" scripts/e1_tinyfat_friction_01_runner.py \
  --manifest "$E1_JOB_MANIFEST" \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --wires-json results/master_thesis/e1_tinyfat_friction_01/metadata/wires.json \
  --worker-count "$E1_WORKER_COUNT" \
  --friction "$E1_FRICTION" \
  --no-write-trace
