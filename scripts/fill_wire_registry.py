#!/usr/bin/env python3
"""Populate wire_registry from archive/agents/complete.

Steps:
  1. Copy old wire_registry agents to archive/agents/old/ (backup), then remove them.
  2. For each of the 15 new agents:
     - Single-run: copy best_checkpoint.everl directly.
     - Two-run: parse both runs' CSVs, pick the run with the higher peak
       `best quality`, copy its best_checkpoint.everl.
  3. Write agent.json and empty __init__.py files.
"""

import csv
import json
import shutil
from pathlib import Path

BASE = Path("/home/hannes-deittert/dev/Uni/master-project/data")
ARCHIVE_COMPLETE = BASE / "archive/agents/complete"
ARCHIVE_OLD = BASE / "archive/agents/old"
WIRE_REG_ROOT = BASE / "wire_registry"

# Old agents to back up and remove from the registry.
OLD_AGENTS = [
    WIRE_REG_ROOT / "steve_default/wire_versions/standard_j/agents/archvar_original_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/tight_j/agents/archvar_steve_tight_j_best",
    WIRE_REG_ROOT / "amplatz_super_stiff/wire_versions/standard_j/agents/archvar_amplatz_superstiff_best",
    WIRE_REG_ROOT / "universal_ii/wire_versions/standard_j/agents/archvar_universalII_best",
]

# Misplaced agents created in the previous (broken) run — all were put under
# steve_default instead of their correct model folder. Remove them so we can
# re-create them in the right place.
MISPLACED_AGENTS = [
    WIRE_REG_ROOT / "steve_default/wire_versions/gentle/agents/archvar_amplatz_gentle_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/gentle/agents/archvar_universalii_gentle_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/standard_j/agents/archvar_amplatz_standard_j_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/standard_j/agents/archvar_universalii_standard_j_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/straight/agents/archvar_amplatz_straight_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/straight/agents/archvar_universalii_straight_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/strong_hook/agents/archvar_amplatz_strong_hook_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/strong_hook/agents/archvar_universalii_strong_hook_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/tight_j/agents/archvar_amplatz_tight_j_best",
    WIRE_REG_ROOT / "steve_default/wire_versions/tight_j/agents/archvar_universalii_tight_j_best",
]

# (model_name, wire_name, new_agent_name, archive_folder_name, is_two_run)
AGENTS = [
    # amplatz_super_stiff
    ("amplatz_super_stiff", "gentle",     "archvar_amplatz_gentle_best",          "complete_amplatz_gentle",        True),
    ("amplatz_super_stiff", "standard_j", "archvar_amplatz_standard_j_best",      "complete_amplatz_standardj",     True),
    ("amplatz_super_stiff", "straight",   "archvar_amplatz_straight_best",        "complete_amplatz_straight",      True),
    ("amplatz_super_stiff", "strong_hook","archvar_amplatz_strong_hook_best",     "complete_amplatz_stronghook",    True),
    ("amplatz_super_stiff", "tight_j",    "archvar_amplatz_tight_j_best",         "complete_amplatz_tightj",        True),
    # steve_default
    ("steve_default",       "gentle",     "archvar_steve_gentle_best",            "complete_steve_gentle",          True),
    ("steve_default",       "standard_j", "archvar_steve_standard_j_best",        "complete_steve_standard_j",      True),
    ("steve_default",       "straight",   "archvar_steve_straight_best",          "complete_steve_straight",        True),
    ("steve_default",       "strong_hook","archvar_steve_strong_hook_best",       "complete_steve_stronghook",      True),
    ("steve_default",       "tight_j",    "archvar_steve_tight_j_best",
     "2026-04-22_070329_crusher_archvar_steve_tight_j_cap150_450_full_nw12_20260422_070328",                        False),
    # universal_ii
    ("universal_ii",        "gentle",     "archvar_universalii_gentle_best",      "complete_universaalii_gentle",   True),
    ("universal_ii",        "standard_j", "archvar_universalii_standard_j_best",  "complete_universalii_standard_j",True),
    ("universal_ii",        "straight",   "archvar_universalii_straight_best",    "complete_universalii_straight",  True),
    ("universal_ii",        "strong_hook","archvar_universalii_strong_hook_best", "complete_universalii_stronghook",True),
    ("universal_ii",        "tight_j",    "archvar_universalii_tight_j_best",
     "2026-04-24_212216_crusher_archvar_universalii_tight_j_cap150_450_full_nw22_20260424_212215",                  False),
]


def _best_quality_from_csv(run_dir: Path) -> float:
    csv_files = [f for f in run_dir.iterdir() if f.suffix == ".csv"]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {run_dir}")
    best = -float("inf")
    with open(csv_files[0], newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = [h.strip() for h in next(reader)]
        next(reader)  # row 2 is config metadata, not data
        col = headers.index("best quality")
        for row in reader:
            if len(row) <= col:
                continue
            try:
                best = max(best, float(row[col]))
            except ValueError:
                continue
    return best


def _pick_best_run(archive_dir: Path):
    """Return (run_dir, best_quality) for the run with the highest best quality."""
    run_dirs = [d for d in archive_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirs in {archive_dir}")
    scored = []
    for d in run_dirs:
        try:
            q = _best_quality_from_csv(d)
            scored.append((q, d))
        except Exception as e:
            print(f"  WARNING: could not score {d.name}: {e}")
    if not scored:
        raise RuntimeError(f"Could not score any runs in {archive_dir}")
    scored.sort(key=lambda x: x[0], reverse=True)
    for q, d in scored:
        print(f"    {q:.4f}  {d.name}")
    return scored[0][1], scored[0][0]


def _create_agent(model: str, wire: str, name: str, best_ckpt_src: Path) -> None:
    wire_versions = WIRE_REG_ROOT / model / "wire_versions"
    agent_dir = wire_versions / wire / "agents" / name
    ckpt_dir = agent_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    (wire_versions / wire / "agents" / "__init__.py").touch()
    (agent_dir / "__init__.py").touch()
    (ckpt_dir / "__init__.py").touch()

    dest = ckpt_dir / "best_checkpoint.everl"
    print(f"  copying best_checkpoint.everl …")
    shutil.copy2(best_ckpt_src, dest)

    agent_json = {
        "checkpoint": str(dest),
        "name": name,
        "run_dir": str(ckpt_dir),
        "source_checkpoint": str(best_ckpt_src),
        "source_agent_json": None,
        "tool": f"{model}/{wire}",
    }
    with open(agent_dir / "agent.json", "w") as f:
        json.dump(agent_json, f, indent=2)

    print(f"  done → {model}/wire_versions/{wire}/agents/{name}")


def _backup_old_agents() -> None:
    ARCHIVE_OLD.mkdir(parents=True, exist_ok=True)
    for old in OLD_AGENTS:
        if not old.exists():
            print(f"  skip (not found): {old.name}")
            continue
        dest = ARCHIVE_OLD / old.name
        if dest.exists():
            print(f"  skip backup (already exists): {dest}")
        else:
            print(f"  backing up {old.name} → archive/old/")
            shutil.copytree(old, dest)
        print(f"  removing {old.name} from wire_registry")
        shutil.rmtree(old)


def _remove_misplaced_agents() -> None:
    for path in MISPLACED_AGENTS:
        if path.exists():
            print(f"  removing misplaced: {path.relative_to(WIRE_REG_ROOT)}")
            shutil.rmtree(path)
        else:
            print(f"  already gone: {path.relative_to(WIRE_REG_ROOT)}")


def main() -> None:
    print("=== Step 1: back up and remove old agents ===")
    _backup_old_agents()

    print("\n=== Step 2: remove misplaced agents from previous run ===")
    _remove_misplaced_agents()

    print("\n=== Step 3: create new agents in correct model folders ===")
    errors = []
    for model, wire, name, archive_source, is_two_run in AGENTS:
        print(f"\n[{model}/{wire}] {name}")
        archive_dir = ARCHIVE_COMPLETE / archive_source
        if not archive_dir.exists():
            msg = f"archive dir not found: {archive_dir}"
            print(f"  ERROR: {msg}")
            errors.append(msg)
            continue

        try:
            if is_two_run:
                print("  comparing runs …")
                best_run, quality = _pick_best_run(archive_dir)
                print(f"  → best run (quality={quality:.4f}): {best_run.name}")
                best_ckpt_src = best_run / "checkpoints" / "best_checkpoint.everl"
            else:
                best_ckpt_src = archive_dir / "checkpoints" / "best_checkpoint.everl"

            if not best_ckpt_src.exists():
                raise FileNotFoundError(f"best_checkpoint.everl not found: {best_ckpt_src}")

            _create_agent(model, wire, name, best_ckpt_src)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(f"[{model}/{wire}] {name}: {e}")

    print("\n=== Done ===")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All 15 agents created successfully.")


if __name__ == "__main__":
    main()
