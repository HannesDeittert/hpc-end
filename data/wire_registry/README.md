# Wire Registry (`data/wire_registry`)

Canonical wire registry for model/version-based tool refs.

## Scope

Current migrated scope: `ArchVarJShaped` into three wire-model families:

- `steve_default`
- `amplatz_super_stiff`
- `universal_ii`

## Layout

```
data/wire_registry/
  <wire_model>/
    model_definition.json
    wire_versions/
      <version>/
        tool.py
        tool_definition.json
        agents/
          <agent>/
            agent.json
            checkpoints/
              *.everl
```

`__pycache__` is runtime-generated and intentionally not tracked.

## Metadata index

`index.json` is a generated inventory snapshot of the canonical tree.

```python
from data.wire_registry import bootstrap_to_disk, load_registry

bootstrap_to_disk()
registry = load_registry()
print(len(registry.models), len(registry.wires), len(registry.agents))
```

## Migration artifacts

- `archvar_inventory_manifest.json`: frozen source inventory and baseline counts.
- `archvar_migration_manifest.json`: copy/parity verification results.
