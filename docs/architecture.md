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
   - Comparison orchestration (`steve_recommender/comparison/`) resolves
     registry candidates into concrete tool+checkpoint pairs.
   - CLI entrypoints: `steve-train`, `steve-eval`, `steve-compare`.

5) **UI** (`steve_recommender/ui/` + `steve_recommender/main.py`)
   - PyQt UI that talks to services only.
   - Entry point: `steve-ui-qt`.

## Storage layout

Canonical wire registry:

```
data/wire_registry/<model>/model_definition.json
data/wire_registry/<model>/wire_versions/<version>/tool.py
data/wire_registry/<model>/wire_versions/<version>/tool_definition.json
data/wire_registry/<model>/wire_versions/<version>/agents/<agent>/agent.json
data/wire_registry/<model>/wire_versions/<version>/agents/<agent>/checkpoints/*.everl
```

Migration details:
`docs/archvar_to_wire_registry_migration.md`

Legacy source (kept for traceability during transition):

```
data/ArchVarJShaped/wires/<wire>/...
```

Current canonical refs:
- `steve_default/default`
- `steve_default/straight_tip`
- `amplatz_super_stiff/default`
- `universal_ii/default`

Historic layout:

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
data/wire_registry/<model>/wire_versions/<version>/tool.py
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

```yaml
use_non_mp_sim: true
force_extraction:
  mode: passive
  required: true
```

Build and export the native monitor plugin before running:

```bash
# Ubuntu: sudo apt install libboost-all-dev cmake build-essential
scripts/build_wall_force_monitor.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN=/.../libSofaWireForceMonitor.so
```

The monitor is injected by the recommender runtime (collector side), so
upstream `third_party/stEVE` sources remain unchanged.
