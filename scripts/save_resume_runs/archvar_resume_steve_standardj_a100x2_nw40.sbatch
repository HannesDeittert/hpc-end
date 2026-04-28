#!/bin/bash -l
#SBATCH --job-name=archvar_resume_steve_standardj_a100x2_nw40
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/home/woody/iwhr/iwhr106h/master-project/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94/slurm_%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

ulimit -n 65535 || ulimit -n 32768 || ulimit -n 8192

module purge
module load python

eval "$(conda shell.bash hook)"
conda activate /home/woody/iwhr/iwhr106h/conda/envs/master-project

cd /home/woody/iwhr/iwhr106h/master-project

export PYTHONPATH=/home/woody/iwhr/iwhr106h/master-project${PYTHONPATH:+:$PYTHONPATH}
export SOFA_ROOT=/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux

export MPLCONFIGDIR=/tmp/mplconfig_${SLURM_JOB_ID:-direct}
mkdir -p "$MPLCONFIGDIR"

source scripts/sofa_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2

RESULTS=/home/woody/iwhr/iwhr106h/master-project/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94
OLD_RUN=2026-04-25_163824_a100x1_archvar_steve_standardj_cap150_450_full_1583390
OLD_RUN_DIR=${RESULTS}/${OLD_RUN}

CKPT=$(find "${OLD_RUN_DIR}/checkpoints" -maxdepth 1 -type f -name 'checkpoint*.everl' | sort -V | tail -1)
SIDECAR="${OLD_RUN_DIR}/checkpoints/latest_replay_buffer.everl"

NW=${NW:-40}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID:-direct_$(date +%Y%m%d_%H%M%S)}}
RUN_NAME=a100x2_archvar_steve_standardj_checked_nw${NW}_resume_${RUN_ID}

hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-direct}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ulimit -n=$(ulimit -n)"
echo "NW=${NW}"
echo "RUN_NAME=${RUN_NAME}"
echo "CKPT=${CKPT}"
echo "SIDECAR=${SIDECAR}"

nvidia-smi

python scripts/archvar_train_tool.py \
  -d cuda:0 \
  -nw "${NW}" \
  -n "${RUN_NAME}" \
  --heatup-steps 500000 \
  --training-steps 20000000 \
  --explore-steps-between-eval 250000 \
  --explore-episodes-between-updates 100 \
  -lr 0.0003218 \
  --hidden 400 400 400 \
  -en 900 \
  -el 1 \
  --train-max-steps 150 \
  --eval-max-steps 450 \
  --save-latest-replay-buffer \
  --resume-from "${CKPT}" \
  --resume-replay-buffer-from "${SIDECAR}" \
  --tool-module data.wire_registry.steve_default.wire_versions.standard_j.tool \
  --tool-class JShaped_Default_StandardJ \
  --results-folder "${RESULTS}"
