"""Module entry point for the standalone eval_v2 replay viewer."""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr))
