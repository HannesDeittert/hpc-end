# Aortic-Arch Anatomy Dataset (bank generation + UI selection)

This repo includes a small, repo-local dataset format to store a **large bank of reproducible aortic arch anatomies** (e.g. 10,000) for evaluation/benchmarking.

The dataset is stored under `results/` (gitignored) and is meant to be:
- generated once (or incrementally in batches),
- browsed/previewed in the UI (centerlines),
- selected and used for evaluation runs (seed + parameters).

## Where it is stored

Default dataset root:

`results/anatomies/aortic_arch/`

Layout:

- `results/anatomies/aortic_arch/index.jsonl`  
  JSONL file (1 line = 1 anatomy record) used by the UI table.
- `results/anatomies/aortic_arch/records/arch_000000/description.json`  
  Parameters to recreate the anatomy via stEVE `AorticArch`.
- `results/anatomies/aortic_arch/records/arch_000000/centerline.npz`  
  Precomputed branch polylines + insertion pose for fast UI preview (no meshing).

Important: We intentionally do **not** generate/store vessel meshes here. The simulator can reconstruct the vessel tree from `arch_type + seed (+ transforms)` during evaluation.

## Generate 10,000 anatomies (in batches)

Prerequisites:

```bash
conda activate master-project
```

Generate 10 batches of 1,000 (append mode):

```bash
for _ in $(seq 1 10); do
  python -m steve_recommender.anatomy.generate_aortic_arch_dataset \
    --resume \
    --count 1000 \
    --dataset-seed 123 \
    --progress-every 100
done
```

Or generate all 10,000 in one run:

```bash
python -m steve_recommender.anatomy.generate_aortic_arch_dataset \
  --count 10000 \
  --dataset-seed 123 \
  --progress-every 200
```

Notes:
- Re-running with `--resume` continues at the next free `arch_XXXXXX` index.
- If a record folder exists but is missing from `index.jsonl` (e.g. after an interrupted run), the generator tries to **repair** the index entry automatically.

## UI usage (preview + evaluation)

1. Start the UI:

```bash
python -m steve_recommender.main
```

2. In the **Evaluate** tab:
   - click **Select Anatomy…**
   - browse the dataset table (fast even for ~10k rows)
   - inspect the **centerline preview**
   - select an anatomy (this sets `arch_type`, `seed`, and transform parameters in the eval config)

3. Add agents (checkpoint + tool) and run the evaluation.
