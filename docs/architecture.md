# Architecture

This repo is organized as a layered, pythonic stack. The goal is to keep
stEVE integration isolated, business logic testable, and the UI thin.

## Layers

1) **Adapters** (`steve_recommender/adapters/`)
   - The only place that imports stEVE packages directly.
   - Use `from steve_recommender.adapters import eve, eve_rl` everywhere else.

2) **Domain** (`steve_recommender/domain/`)
   - Dataclasses that describe models, runs, and configs.
   - Example: `TrainingConfig`, `ModelInfo`, `WireInfo`.

3) **Services** (`steve_recommender/services/`)
   - Pythonic entrypoints used by CLI or UI.
   - Examples: `training_service.run_training`, `evaluation_service.run`, `library_service.list_models`.

4) **Pipelines** (`steve_recommender/rl/` and `steve_recommender/evaluation/`)
   - Core training and evaluation logic.
   - CLI entrypoints still live here (`steve-train`, `steve-eval`).

5) **UI** (`steve_recommender/ui_nicegui/`)
   - NiceGUI UI that talks to services only.
   - Entry point: `steve-ui`.

## Storage layout

```
data/<model>/model_definition.json
data/<model>/wires/<wire>/tool.py
data/<model>/wires/<wire>/tool_definition.json
data/<model>/wires/<wire>/agents/<agent>/agent.json
results/paper_runs/<run>/main.log
results/paper_runs/<run>.csv
results/eval_runs/<run>/summary.csv
```

## Extension points

### 1) Add a new device class (stEVE Device subclass)

Option A: **Project library class** (recommended for reusable templates)

```
steve_recommender/devices/<your_device>.py
```

```python
from dataclasses import dataclass
from steve_recommender.adapters import eve

Device = eve.intervention.device.device.Device

@dataclass
class MyWire(Device):
    name: str = "MyWire"

    def __post_init__(self) -> None:
        super().__init__(...)
```

Option B: **User-generated wire** (UI output)

```
data/<model>/wires/<wire>/tool.py
```

The Qt wizard already generates this file, but future UIs should use the same
pattern and import stEVE through the adapter.

### 2) Extract extra data during evaluation

1. Add a collector in `steve_recommender/evaluation/info_collectors.py`
2. Register it inside `steve_recommender/evaluation/pipeline.py`:

```python
info = eve.info.Combination([..., MyCollector(intervention)])
```

3. Save arrays in the `.npz` bundle in the same file.

If you need access to SOFA internals, enable:

```
use_non_mp_sim: true
```

in the evaluation config so the scene graph is accessible from Python.

## NiceGUI

Install the optional UI dependency and start the app:

```
pip install -e .[ui]
steve-ui
```
