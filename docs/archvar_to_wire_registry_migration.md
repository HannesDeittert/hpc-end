# ArchVarJShaped to Wire Registry Migration

## Mapping

| Legacy ref | Canonical ref |
|---|---|
| `ArchVarJShaped/JShaped_Default` | `steve_default/default` |
| `ArchVarJShaped/JShaped_Default_StraightTip` | `steve_default/straight_tip` |
| `ArchVarJShaped/j_shaped_AmplatzSuperStiff` | `amplatz_super_stiff/default` |
| `ArchVarJShaped/j_shaped_UniversalII` | `universal_ii/default` |

## Phases Executed

1. Inventory + baseline freeze to:
   - `data/wire_registry/archvar_inventory_manifest.json`
2. Canonical model/version skeleton + metadata creation under:
   - `data/wire_registry/<model>/wire_versions/<version>/...`
3. Full checkpoint copy for all ArchVar agents:
   - 4 agents
   - 324 checkpoint files
4. Runtime path cutover to canonical model/version refs.
5. Documentation and package metadata updates.

## Validation Snapshot

See:
- `data/wire_registry/archvar_migration_manifest.json`
- `data/wire_registry/index.json`

Expected parity:
- canonical versions: 4
- canonical agents: 4
- canonical checkpoints: 324
- size parity: true
- best checkpoint hash parity: true

## Rollback Notes

Legacy source files remain in `data/ArchVarJShaped/wires/*`.
If rollback is required:

1. revert runtime/storage path changes
2. switch tool refs back to legacy names
3. keep migrated `data/wire_registry/*` as read-only audit artifacts
