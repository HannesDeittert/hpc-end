"""Generate a small aortic-arch dataset batch."""

from __future__ import annotations

from pathlib import Path

from steve_recommender.anatomy.aortic_arch_dataset import (
    generate_aortic_arch_records,
    load_aortic_arch_dataset,
    next_record_index,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset = load_aortic_arch_dataset(repo_root / "results" / "anatomies" / "aortic_arch")
    start_index = next_record_index(dataset)
    records = generate_aortic_arch_records(
        dataset=dataset,
        start_index=start_index,
        count=10,
        dataset_seed=123,
        arch_types=["I", "II", "IV"],
        write_centerlines=True,
        overwrite=False,
        progress_every=5,
    )
    print(f"[example] generated {len(records)} records at {dataset.root}")


if __name__ == "__main__":
    main()
