#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/hannes-deittert/dev/Uni/master-project${PYTHONPATH:+:$PYTHONPATH}
source scripts/sofa_env.sh

RESULTS=/home/hannes-deittert/dev/Uni/master-project/results/train_v2_smoke_cuda_repro
mkdir -p "$RESULTS"

python -u -m steve_recommender.train_v2 train \
  --name smoke_cuda_repro_force_relu \
  --tool custom \
  --tool-module data.wire_registry.amplatz_super_stiff.wire_versions.standard_j.tool \
  --tool-class JShaped_AmplatzSuperStiff_StandardJ \
  --trainer-device cuda:0 \
  --worker-count 1 \
  --heatup-steps 8 \
  --training-steps 28 \
  --eval-every 10 \
  --eval-episodes 2 \
  --explore-episodes-between-updates 1 \
  --batch-size 4 \
  --train-max-steps 5 \
  --eval-max-steps 5 \
  --reward-profile default_plus_normal_force_penalty \
  --force-alpha 0.1 \
  --force-beta 1.0 \
  --force-penalty-mode relu_threshold \
  --force-threshold 0.8 \
  --force-region whole_wire \
  --output-root "$RESULTS" \
  --no-preflight
