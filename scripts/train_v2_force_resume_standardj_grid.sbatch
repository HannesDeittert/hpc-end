#!/bin/bash -l
#SBATCH --job-name=train_v2_force_resume_standardj
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/woody/iwhr/iwhr106h/master-project/results/train_v2_force_resume_standardj/logs/slurm-%A_%a.out
#SBATCH --error=/home/woody/iwhr/iwhr106h/master-project/results/train_v2_force_resume_standardj/logs/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

PROJECT_ROOT=${PROJECT_ROOT:-/home/woody/iwhr/iwhr106h/master-project}
CONDA_ENV=${CONDA_ENV:-/home/woody/iwhr/iwhr106h/conda/envs/master-project}
SOFA_ROOT=${SOFA_ROOT:-/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/results/train_v2_force_resume_standardj}

mkdir -p "${OUTPUT_ROOT}/logs"

module purge
module load python

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SOFA_ROOT
source scripts/sofa_env.sh

export STEVE_WALL_FORCE_MONITOR_PLUGIN="${STEVE_WALL_FORCE_MONITOR_PLUGIN:-${PROJECT_ROOT}/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so}"
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID:-direct}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2

# Grid: 2 non-force standard-J checkpoints x 3 weak force coefficients.
WIRE_KEYS=(steve_default amplatz_super_stiff)
WIRE_LABELS=(steve amplatz)
TOOL_MODULES=(
  data.wire_registry.steve_default.wire_versions.standard_j.tool
  data.wire_registry.amplatz_super_stiff.wire_versions.standard_j.tool
)
TOOL_CLASSES=(
  JShaped_Default_StandardJ
  JShaped_AmplatzSuperStiff_StandardJ
)
CHECKPOINTS=(
  "${PROJECT_ROOT}/data/wire_registry/steve_default/wire_versions/standard_j/agents/archvar_steve_standard_j_best/checkpoints/best_checkpoint.everl"
  "${PROJECT_ROOT}/data/wire_registry/amplatz_super_stiff/wire_versions/standard_j/agents/archvar_amplatz_standard_j_best/checkpoints/best_checkpoint.everl"
)
FORCE_ALPHAS=(0.001 0.005 0.01)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
WIRE_IDX=$(( TASK_ID / ${#FORCE_ALPHAS[@]} ))
ALPHA_IDX=$(( TASK_ID % ${#FORCE_ALPHAS[@]} ))

WIRE_KEY=${WIRE_KEYS[$WIRE_IDX]}
WIRE_LABEL=${WIRE_LABELS[$WIRE_IDX]}
TOOL_MODULE=${TOOL_MODULES[$WIRE_IDX]}
TOOL_CLASS=${TOOL_CLASSES[$WIRE_IDX]}
CKPT=${CHECKPOINTS[$WIRE_IDX]}
FORCE_ALPHA=${FORCE_ALPHAS[$ALPHA_IDX]}
ALPHA_TAG=$(printf '%s' "${FORCE_ALPHA}" | sed 's/^0\.//; s/\./p/g')

WORKER_COUNT=${WORKER_COUNT:-24}
HEATUP_STEPS=${HEATUP_STEPS:-50000}
TRAINING_STEPS=${TRAINING_STEPS:-2000000}
EVAL_EVERY=${EVAL_EVERY:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-20}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-150}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-450}
FORCE_THRESHOLD=${FORCE_THRESHOLD:-0.8}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-direct}}
RUN_NAME="force_resume_${WIRE_LABEL}_standard_j_relu_a${ALPHA_TAG}_${RUN_TAG}_task${TASK_ID}"

if [[ ! -f "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 2
fi

hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-direct}"
echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "WIRE_KEY=${WIRE_KEY}"
echo "TOOL_MODULE=${TOOL_MODULE}"
echo "TOOL_CLASS=${TOOL_CLASS}"
echo "CKPT=${CKPT}"
echo "FORCE_ALPHA=${FORCE_ALPHA}"
echo "FORCE_THRESHOLD=${FORCE_THRESHOLD}"
echo "WORKER_COUNT=${WORKER_COUNT}"
echo "HEATUP_STEPS=${HEATUP_STEPS}"
echo "TRAINING_STEPS=${TRAINING_STEPS}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "EVAL_EPISODES=${EVAL_EPISODES}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

nvidia-smi || true

python -m steve_recommender.train_v2 train \
  --name "${RUN_NAME}" \
  --tool "${WIRE_KEY}/standard_j" \
  --tool-module "${TOOL_MODULE}" \
  --tool-class "${TOOL_CLASS}" \
  --reward-profile default_plus_normal_force_penalty \
  --force-penalty-mode relu_threshold \
  --force-alpha "${FORCE_ALPHA}" \
  --force-beta 0.0 \
  --force-threshold "${FORCE_THRESHOLD}" \
  --force-region whole_wire \
  --trainer-device cuda:0 \
  --worker-device cpu \
  --replay-device cpu \
  --output-root "${OUTPUT_ROOT}" \
  --worker-count "${WORKER_COUNT}" \
  --heatup-steps "${HEATUP_STEPS}" \
  --training-steps "${TRAINING_STEPS}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --explore-episodes-between-updates 100 \
  --learning-rate 0.0003218 \
  --hidden 400 400 400 \
  --embedder-nodes 900 \
  --embedder-layers 1 \
  --train-max-steps "${TRAIN_MAX_STEPS}" \
  --eval-max-steps "${EVAL_MAX_STEPS}" \
  --resume-from "${CKPT}" \
  --save-latest-replay-buffer
