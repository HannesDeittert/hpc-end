# steve-recommender

> **Purpose**\
> Reinforcement‑learning–based framework to **recommend the optimal guidewire** for a given vessel geometry.\
> Builds on the open‑source *stEVE* simulator suite (SOFA backend) and the "Recurrent guidewire navigation" paper implementation.

---

## Table of Contents

1. [Project layout](#project-layout)
2. [Getting Started](#quick-start)
3. [Install SOFA with SofaPython3 & BeamAdapter](#installation-detailed)
4. [Roadmap](#roadmap)

---

## Project layout

```
master-project/
├── steve_recommender/          # package code
├── examples/                   # runnable examples
├── experiments/                # notebooks, adhoc scripts
├── docs/                       # images, thesis diagrams, reports
├── data/                       # user-created models & wires (UI output)
├── results/                    # generated runs (gitignored)
├── .gitignore
├── CONTRIBUTING.md             # guideline for contributing and working with the project
└── README.md
```

---

## Getting started

1. **Follow the instructions to install SOFA with SofaPython3 and BeamAdapter**
2. **Clone the repo**
```bash
$ git clone https://github.com/HannesDeittert/steve-recommender.git
$ cd steve-recommender
```
3. **Create & activate conda env (Python 3.8)**
```bash
$ mamba create -n env_name python=3.8 -y
$ conda activate env_name
```
4. **Install the package (pulls stEVE dependencies via git pins)**
```bash
$ pip install -e .
```
5. **Run smoke test**
```bash
$ python -m steve_recommender.rl.smoke_train --tool <model>/<wire> --device cpu
```

---
## Models & wires

The UI stores your created assets under `data/`:

- `data/<model>/model_definition.json`
- `data/<model>/wires/<wire>/tool.py`
- `data/<model>/wires/<wire>/tool_definition.json`
- `data/<model>/wires/<wire>/agents/` (optional training artifacts)

---
## Training (paper architecture)

This repo provides its own training entrypoints (outside upstream stEVE repos) that mirror the stEVE_training scripts but load your stored wires from `data/`.

- Docs: `docs/training_pipeline.md`
- Entrypoints: `steve-train` (multi-worker) and `python -m steve_recommender.rl.train_paper_arch_single` (single agent / debug)
- Helper scripts: `scripts/sofa_env.sh`, `scripts/train_paper.sh`, `scripts/monitor_paper.sh`

---
## Evaluation (benchmark pipeline)

Repo-local evaluation pipeline to benchmark **multiple trained agents** (each with its own tool + checkpoint) on a fixed anatomy.

- Docs: `docs/evaluation_pipeline.md`
- CLI: `steve-eval --config docs/eval_example.yml`
- Outputs: `results/eval_runs/<timestamp>_<name>/summary.csv` + `trials/*.npz`

---
## UI (NiceGUI, optional)

NiceGUI UI for browsing models/wires, launching training, and running evaluation.

```bash
$ pip install -e .[ui]
$ steve-ui
```

---
## Install SOFA with SofaPython3 & BeamAdapter

You have **two ways** to get SOFA (incl. SofaPython3 and BeamAdapter) up and running. *For almost everyone, downloading the official binaries is the fastest and most reliable path.*

###  A) Download pre‑built binaries — **recommended**

1. **Grab the archive** (≤ v23.06): [https://github.com/sofa-framework/sofa/releases/tag/v23.06.00](https://github.com/sofa-framework/sofa/releases/tag/v23.06.00)
2. **Extract** it somewhere convenient, e.g. `$HOME/opt/SOFA_v23.06.00_Linux`.
3. **Install OS dependencies**
4. **Set environment variables** – add this to `.bashrc` **or** run it once per shell:
   ```bash
   export SOFA_ROOT=$HOME/opt/SOFA_v23.06.00_Linux/SOFA_v23.06.00_Linux
   export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
   ```
   \* `SOFA_ROOT` is mandatory so SOFA can locate its plugins.\
   \* `PYTHONPATH` tells Python where the SofaPython3 packages live.\

###  B) Compile SOFA yourself — **advanced / not tested by maintainer**

If you need a custom build (other branch, debug symbols, ARM Linux…), follow the compile guide in the stEVE fork: [https://github.com/lkarstensen/stEVE#build-sofa-yourself](https://github.com/lkarstensen/stEVE#build-sofa-yourself)

---

## Roadmap

-

---

> **Maintainer**  Hannes Deittert\
> **Last update**  2025‑07‑21
