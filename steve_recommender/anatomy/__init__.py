"""Anatomy generation and storage helpers.

Goal:
- Make anatomies reproducible (parameter + seed based)
- Allow generating large banks of anatomies for evaluation/benchmarking
- Keep the data optional to commit (store under `results/`)
"""

from .aortic_arch_dataset import (
    AorticArchDataset,
    AorticArchRecord,
    load_aortic_arch_dataset,
)

__all__ = ["AorticArchDataset", "AorticArchRecord", "load_aortic_arch_dataset"]
