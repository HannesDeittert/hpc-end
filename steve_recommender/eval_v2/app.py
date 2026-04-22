from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_qt_plugin_paths() -> None:
    """Prefer the active conda environment Qt plugins over external paths.

    Some local setups export `QT_QPA_PLATFORM_PLUGIN_PATH` (for example from Sofa),
    and OpenCV can also inject Qt plugin paths. This forces the launcher to use
    the current Python environment's Qt plugins so PyQt5 can initialize reliably.
    """

    env_root = Path(sys.executable).resolve().parents[1]
    plugins_root = env_root / "plugins"
    platforms_dir = plugins_root / "platforms"
    if platforms_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
    if plugins_root.exists():
        os.environ["QT_PLUGIN_PATH"] = str(plugins_root)


def _apply_theme() -> None:
    """Apply premium dark theme when qdarktheme is available."""
    try:
        import qdarktheme
    except ImportError:
        return

    qdarktheme.setup_theme()


def main() -> int:
    _configure_qt_plugin_paths()

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    _apply_theme()

    from .ui_main import ClinicalMainWindow

    main_window = ClinicalMainWindow()
    main_window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
