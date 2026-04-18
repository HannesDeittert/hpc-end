#!/usr/bin/env python3
"""Promote a backlog wire into the active wires registry.

Example:
  python scripts/promote_backlog_wire.py \
    --model ArchVarJShaped \
    --wire j_shaped_AmplatzSuperStiff \
    --mode move
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_package_marker(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").touch(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name under data/<model>")
    parser.add_argument(
        "--wire", required=True, help="Backlog wire directory name under backlog_wires/"
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="copy: keep backlog entry, move: remove backlog entry after copy",
    )
    args = parser.parse_args()

    root = _repo_root()
    model_dir = root / "data" / args.model
    backlog_root = model_dir / "backlog_wires"
    wires_root = model_dir / "wires"
    src = backlog_root / args.wire
    dst = wires_root / args.wire

    if not src.exists():
        print(f"[error] backlog wire not found: {src}", file=sys.stderr)
        return 1
    if not (src / "tool.py").exists():
        print(f"[error] missing tool.py in backlog wire: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"[error] destination already exists: {dst}", file=sys.stderr)
        return 1

    _ensure_package_marker(model_dir)
    _ensure_package_marker(wires_root)

    shutil.copytree(src, dst)
    (dst / "__init__.py").touch(exist_ok=True)

    if args.mode == "move":
        shutil.rmtree(src)

    print(f"[ok] promoted {args.model}/{args.wire} -> {dst}")
    print(f"[ok] mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
