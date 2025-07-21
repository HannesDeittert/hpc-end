#!/usr/bin/env python3
"""tool_compare main

Führt einen kurzen Selbsttest aus:
1. Importiert das editierbar installierte *eve*-Paket (kommt aus third_party/stEVE).
2. Ruft das mitgelieferte Beispielskript *examples/function_check.py* auf
   und gibt dessen Ausgabe direkt durch.

Das Skript kann von überall in der aktivierten Umgebung gestartet werden,
solange *eve* via `pip install -e ./third_party/stEVE` verlinkt ist.
"""
from __future__ import annotations

import runpy
from pathlib import Path
import sys


def main() -> None:
    """Importiere *eve* und führe den Beispiel‑Self‑Check aus."""

    # 1️⃣  Sicherstellen, dass *eve* importierbar ist
    try:
        import eve  # noqa: F401  # pylint: disable=unused-import
    except ModuleNotFoundError as exc:
        print(
            "[ERROR] Das Paket 'eve' ist nicht installiert oder nicht im Python‑Pfad.\n"
            "Führe zuerst `pip install -e ./third_party/stEVE` im Projekt‑Root aus.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # 2️⃣  Pfad zum Beispielskript auflösen (relativ zum Projekt‑Root)
    project_root = Path(__file__).resolve().parents[2]  # ~/dev/Uni/master-project
    examples_script = project_root / "third_party" / "stEVE" / "examples" / "function_check.py"

    if not examples_script.exists():
        print(
            f"[ERROR] Beispielskript nicht gefunden: {examples_script}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("[INFO] *eve* erfolgreich importiert. Starte function_check …\n")

    # 3️⃣  Script ausführen, als wäre es via CLI aufgerufen
    runpy.run_path(str(examples_script), run_name="__main__")


if __name__ == "__main__":
    main()
