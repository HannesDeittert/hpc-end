from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Dict, Iterator, Type

import torch
import torch.optim.lr_scheduler as lr_scheduler


@contextmanager
def legacy_checkpoint_load_context() -> Iterator[None]:
    """Temporarily default `torch.load(..., weights_only=False)`.

    PyTorch >= 2.6 changed the default to `weights_only=True`, which breaks
    loading older stEVE/eve_rl checkpoints that contain full pickled objects.
    For trusted local checkpoints we explicitly revert to the legacy default
    while loading.
    """

    original_torch_load = torch.load
    patched_scheduler_inits: Dict[Type[object], object] = {}

    base_scheduler_types = []
    for name in ("LRScheduler", "_LRScheduler"):
        base = getattr(lr_scheduler, name, None)
        if isinstance(base, type):
            base_scheduler_types.append(base)
    base_scheduler_types = tuple(base_scheduler_types)

    def _compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    def _patch_scheduler_verbose_kwargs() -> None:
        if not base_scheduler_types:
            return
        for _, cls in vars(lr_scheduler).items():
            if not isinstance(cls, type):
                continue
            if not issubclass(cls, base_scheduler_types):
                continue
            init = getattr(cls, "__init__", None)
            if init is None:
                continue
            try:
                sig = inspect.signature(init)
            except Exception:
                continue
            if "verbose" in sig.parameters:
                continue

            def _wrapped_init(self, *args, __orig=init, **kwargs):
                kwargs.pop("verbose", None)
                return __orig(self, *args, **kwargs)

            patched_scheduler_inits[cls] = init
            cls.__init__ = _wrapped_init  # type: ignore[assignment]

    torch.load = _compat_torch_load  # type: ignore[assignment]
    _patch_scheduler_verbose_kwargs()
    try:
        yield
    finally:
        for cls, init in patched_scheduler_inits.items():
            cls.__init__ = init  # type: ignore[assignment]
        torch.load = original_torch_load  # type: ignore[assignment]
