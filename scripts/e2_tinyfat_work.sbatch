#!/bin/bash -l
#SBATCH --job-name=e2_tinyfat_work
#SBATCH --partition=work
#SBATCH --cpus-per-task=32
#SBATCH --hint=multithread
#SBATCH --time=24:00:00
#SBATCH --array=0-9999
#SBATCH --output=/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e2_tinyfat/logs/slurm-%A_%a.out
#SBATCH --error=/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e2_tinyfat/logs/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

cd /home/hannes-deittert/dev/Uni/master-project
module load python
eval "$(conda shell.bash hook)"
conda activate "${E2_CONDA_ENV:-/home/woody/iwhr/iwhr106h/conda/envs/master-project}"
export SOFA_ROOT="${SOFA_ROOT:-/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux}"
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="${STEVE_WALL_FORCE_MONITOR_PLUGIN:-$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export E2_JOB_MANIFEST="/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e2_tinyfat/metadata/job_manifest_work.json"
export E2_WORKER_COUNT="60"
export E2_FRICTION="0.1"

exec "$PYTHON_BIN" experiments/master-thesis/run_e2_cell.py   --manifest "$E2_JOB_MANIFEST"   --array-index "$SLURM_ARRAY_TASK_ID"   --worker-count "$E2_WORKER_COUNT"   --wires-json "/home/hannes-deittert/dev/Uni/master-project/results/master_thesis/e2_tinyfat/metadata/wires.json"
