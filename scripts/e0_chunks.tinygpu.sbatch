#!/bin/bash -l
#SBATCH --job-name=e0_chunks
#SBATCH --partition=work
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-3
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

cd /home/hannes-deittert/dev/Uni/master-project
export SOFA_ROOT=/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so"

CHUNK_ID="${SLURM_ARRAY_TASK_ID}"
CHUNK_FILE="results/experimental_prep/sample_12_chunk_${CHUNK_ID}.json"
OUTPUT_ROOT="results/master_thesis/e0_chunks/chunk_${CHUNK_ID}"

RUNS_PER_ANATOMY=3
TRIAL_COUNT=100
MAX_EPISODE_STEPS=1000
WORKER_COUNT=29
BASE_SEED_START=$((123 + 200 * CHUNK_ID * 3 * RUNS_PER_ANATOMY))
TARGET_SEED_START=$((9000 + CHUNK_ID * 3 * RUNS_PER_ANATOMY))

mkdir -p "$OUTPUT_ROOT"

SAMPLE_JSON="$CHUNK_FILE" OUTPUT_ROOT="$OUTPUT_ROOT" RUNS_PER_ANATOMY="$RUNS_PER_ANATOMY" TRIAL_COUNT="$TRIAL_COUNT" MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" WORKER_COUNT="$WORKER_COUNT" BASE_SEED_START="$BASE_SEED_START" TARGET_SEED_START="$TARGET_SEED_START" bash experiments/master-thesis/run_e0_experiment.sh
