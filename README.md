# gw-tool-recommender

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
├── third_party/                # ⤵ five git submodules (exact upstream commits)
│   ├── stEVE/
│   ├── stEVE_bench/
│   ├── stEVE_rl/
│   ├── stEVE_training/
│   └── aortic_arch_generator/
├── dev/                        # source code (tool‑compare logic, XAI, etc.)
│   ├── tool_compare/
├── experiments/                # notebooks, adhoc scripts
├── docs/                       # images, thesis diagrams, reports
├── data/                       # user-created models & wires (UI output)
├── .gitignore
├── .gitmodules                 # records the submodule URLs & paths
├── CONTRIBUTING.md             # guideline for contributing and working with the project
└── README.md                   
```

---

## Getting started

1. **Fowllow the instructions to install SOFA with SofaPython3 and BeamAdapter**
2. **Clone incl. submodules**
```bash
# clone incl. submodules
$ git clone --recursive https://github.com/HannesDeittert/gw-tool-recommender.git
$ cd gw-tool-recommender
```
3. **Create & activate conda env (Python 3.8)**
```bash
$ mamba create -n env_name python=3.8 -y
$ conda activate env_name
```
4. **Create & activate conda env (Python 3.8)**
```bash
# editable‑install of all stEVE packages
$ pip install -e third_party/stEVE
$ pip install -e third_party/stEVE_bench
$ pip install -e third_party/stEVE_rl
$ pip install -e third_party/aortic_arch_generator
$ pip install -e third_party/stEVE_training/eve
$ pip install -e third_party/stEVE_training/eve_bench
$ pip install -e third_party/stEVE_training/eve_rl
```
4. **Run smoke test**
```bash
$ python third_party/stEVE/examples/function_check.py
```

---
## Models & wires

The UI stores your created assets under `data/`:

- `data/<model>/model_definition.json`
- `data/<model>/wires/<wire>/tool.py`
- `data/<model>/wires/<wire>/tool_definition.json`
- `data/<model>/wires/<wire>/agents/` (optional training artifacts)

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
