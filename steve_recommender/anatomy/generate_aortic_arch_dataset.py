from __future__ import annotations

import argparse

from steve_recommender.anatomy.aortic_arch_dataset import (
    generate_aortic_arch_records,
    load_aortic_arch_dataset,
    next_record_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a bank of AorticArch anatomies")
    parser.add_argument("--output", default=None, help="Dataset root folder (default: results/anatomies/aortic_arch)")
    parser.add_argument("--start-index", type=int, default=0, help="First record index to generate (default: 0)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-detect the next free index from the dataset folder and append.",
    )
    parser.add_argument("--count", type=int, default=100, help="How many records to generate in this run")
    parser.add_argument("--dataset-seed", type=int, default=123, help="Base seed controlling the dataset content")
    parser.add_argument(
        "--arch-types",
        nargs="*",
        default=["I", "II", "IV", "V", "VI", "VII"],
        help="Subset of arch types to sample from (default: common stEVE types)",
    )
    parser.add_argument(
        "--no-centerlines",
        action="store_true",
        help="Skip writing centerline.npz (faster, but UI preview will be limited).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing record folders (default: skip existing).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N generated records (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_aortic_arch_dataset(args.output)
    start_index = int(args.start_index)
    if args.resume:
        start_index = next_record_index(dataset)
    records = generate_aortic_arch_records(
        dataset=dataset,
        start_index=start_index,
        count=args.count,
        dataset_seed=args.dataset_seed,
        arch_types=args.arch_types,
        write_centerlines=not args.no_centerlines,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(f"[aortic_arch_dataset] wrote {len(records)} records to: {dataset.root}")
    print(f"[aortic_arch_dataset] index: {dataset.index_path}")
    print(f"[aortic_arch_dataset] records: {dataset.records_dir}")


if __name__ == "__main__":
    main()
