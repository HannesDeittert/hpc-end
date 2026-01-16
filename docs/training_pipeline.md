# Training Pipeline (stEVE-style, repo-local)

This repo trains a **SAC + LSTM (paper architecture)** agent on the **stEVE/SOFA** simulator, but keeps the entrypoints in our own package (`steve_recommender.rl`) with `steve-train` as the main CLI while using the stEVE libraries as dependencies.

## 1) Prerequisites

- A working conda env (Python 3.8) with PyTorch + CUDA if you want GPU training.
- SOFA binaries installed (incl. SofaPython3 + BeamAdapter).
- The repo package installed (`pip install -e .`) to get `steve-train`.

The SOFA Python bindings need:
- `SOFA_ROOT` to locate plugins
- `PYTHONPATH` to include SofaPython3 site-packages
- `LD_LIBRARY_PATH` to include your conda env `lib/` so `libpython3.8.so.1.0` can be resolved

The helper script `scripts/sofa_env.sh` sets these for the current shell.

## 2) Data layout (models + wires)

Created wires are stored under `data/`:

- `data/<model>/model_definition.json`
- `data/<model>/wires/<wire>/tool.py`
- `data/<model>/wires/<wire>/tool_definition.json`
- `data/<model>/wires/<wire>/agents/` (optional artifacts)

Wires are referenced as `<model>/<wire>`, e.g. `TestModel_StandardJ035/StandardJ035_PTFE`.

## 3) Quick environment sanity checks

Activate your env, set SOFA vars, then run a quick import check:

```bash
conda activate master-project
source scripts/sofa_env.sh

python -c "import eve, eve_rl; print('eve', eve.__file__); print('eve_rl', eve_rl.__file__)"
```

## 4) Start training (paper architecture, multi-worker)

Recommended: run with `nohup` so it survives terminal closes/logouts.

```bash
conda activate master-project
bash scripts/train_paper.sh \
  --tool TestModel_StandardJ035/StandardJ035_PTFE \
  --name paper_standardj \
  --device cuda \
  --workers 16
```

Outputs:
- `results/paper_runs/nohup_<name>.log` (stdout + early paths)
- `results/paper_runs/<timestamp>_<name>/main.log` (structured logger)
- `results/paper_runs/<timestamp>_<name>/checkpoints/*.everl`
- `results/paper_runs/<timestamp>_<name>.csv`

Monitor:
```bash
bash scripts/monitor_paper.sh --name paper_standardj
```

Stop (graceful):
```bash
kill -TERM "$(cat results/paper_runs/paper_standardj.pid)"
```

## 5) Resume from a checkpoint

```bash
conda activate master-project
bash scripts/train_paper.sh \
  --tool TestModel_StandardJ035/StandardJ035_PTFE \
  --name paper_standardj_resume \
  --device cuda \
  --workers 16 \
  --resume-from results/paper_runs/<run_folder>/checkpoints/checkpointXXXX.everl
```

## 6) UI progress (TensorBoard: losses, reward, quality)

Install TensorBoard (once):

```bash
conda activate master-project
pip install tensorboard
```

Start training with TensorBoard logging enabled:

```bash
conda activate master-project
bash scripts/train_paper.sh \
  --tool TestModel_StandardJ035/StandardJ035_PTFE \
  --name paper_standardj \
  --device cuda \
  --workers 16 \
  --tensorboard
```

Then launch TensorBoard for the latest run folder:

```bash
latest=$(ls -td results/paper_runs/*paper_standardj*/ 2>/dev/null | head -n 1)
tensorboard --logdir "$latest/tb" --port 6006
```

## Notes on performance

- SOFA simulation is typically CPU-bound; GPU usage can look low even when training is working.
- Increasing workers increases CPU load and environment throughput; it can also make eval timeouts more likely on unstable systems.
