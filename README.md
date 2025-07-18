# gw-tool-recommender

> **Purpose**\
> Reinforcement‑learning–based framework to **recommend the optimal guidewire** for a given vessel geometry.\
> Builds on the open‑source *stEVE* simulator suite (SOFA backend) and the "Recurrent guidewire navigation" paper implementation.

---

## Table of Contents

1. [Project layout](#project-layout)
2. [Quick start](#quick-start)
3. [Installation (detailed)](#installation-detailed)
4. [Running smoke tests](#running-smoke-tests)
5. [Submodules explained](#submodules-explained)
6. [.gitignore rationale](#gitignore-rationale)
7. [Roadmap](#roadmap)

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
├── experiments/                # notebooks, adhoc scripts
├── docs/                       # images, thesis diagrams, reports
├── .gitignore
├── .gitmodules                 # records the submodule URLs & paths
└── README.md                   
```

---

## Quick start

```bash
# clone incl. submodules
$ git clone --recursive https://github.com/HannesDeittert/gw-tool-recommender.git
$ cd gw-tool-recommender

# create & activate conda env (Python 3.8)
$ mamba create -n env_name python=3.8 -y
$ conda activate env_name

# build or install SOFA (with SofaPython3 + BeamAdapter) **against this env**
#  – see section “Installation (detailed)”.

# editable‑install of all stEVE packages
$ pip install -e third_party/stEVE
$ pip install -e third_party/stEVE_bench
$ pip install -e third_party/stEVE_rl
$ pip install -e third_party/aortic_arch_generator
$ pip install -e third_party/stEVE_training/eve
$ pip install -e third_party/stEVE_training/eve_bench
$ pip install -e third_party/stEVE_training/eve_rl

# run smoke test
$ python third_party/stEVE/examples/function_check.py
```

---

## Installation (detailed)

1. **Create Conda environment**\
   `mamba create -n env_name python=3.8 -y && conda activate steve38`

2. **Build SOFA** (preferred) or install binaries.

   ```bash
   mkdir -p ~/opt/sofa/{src,build}
   git clone -b v23.12.00 https://github.com/sofa-framework/sofa.git ~/opt/sofa/src
   cmake -S ~/opt/sofa/src -B ~/opt/sofa/build \
     -D SOFA_FETCH_SOFAPYTHON3=ON -D SOFA_FETCH_BEAMADAPTER=ON \
     -D Python_EXECUTABLE=$(which python) \
     -D SP3_LINK_TO_USER_SITE=True \
     -D SP3_PYTHON_PACKAGES_LINK_DIRECTORY=$(python - <<'EOF'
   ```

import site,sys; print(site.getsitepackages()[0]) EOF ) cmake --build \~/opt/sofa/build --target install -j\$(nproc)

````
> The `SP3_LINK_TO_USER_SITE` flag automatically links SofaPython3 into the active conda env.

3. **Editable‑install packages** (see Quick start).

4. **PyCharm**  
* Open the project root.  
* Set interpreter to **conda env `steve38`**.  
* Mark `dev/` as *Source Root* for your own code.

---

## Running smoke tests
```bash
# core simulator
python third_party/stEVE/examples/function_check.py
# benchmark envs
python third_party/stEVE_bench/examples/function_check.py
# rl framework
python third_party/stEVE_rl/examples/function_check.py
````

A successful run prints `TEST PASSED` and exits without exception.

---

## Roadmap

-

---

> **Maintainer**  Hannes Deittert\
> **Last update**  2025‑07‑18

