#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


def _sanitize_qt_environment() -> None:
    """Avoid OpenCV/SOFA Qt plugin paths shadowing PyQt's xcb plugin."""

    os.environ.pop("QT_PLUGIN_PATH", None)
    current_platform_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    conda_platforms = Path(sys.prefix) / "plugins" / "platforms"
    if conda_platforms.exists() and (
        not current_platform_path
        or "cv2/qt/plugins" in current_platform_path
        or current_platform_path.endswith("/sofa/lib")
        or current_platform_path.endswith("/SOFA/lib")
    ):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(conda_platforms)


_sanitize_qt_environment()

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review E2 anatomy targets in a Qt/PyVista GUI")
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument(
        "--review-json",
        type=Path,
        default=Path("results/master_thesis/e2_target_review/e2_target_review_gui.json"),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sample_records_by_id(targets_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sample_json = Path(str(targets_payload.get("source_sample_json", "")))
    if not sample_json.exists():
        raise FileNotFoundError(f"source_sample_json does not exist: {sample_json}")
    sample_payload = _load_json(sample_json)
    return {
        str(item["record_id"]): dict(item)
        for item in sample_payload.get("selected_anatomies", [])
        if item.get("record_id")
    }


def _load_centerline_branches(path: Path) -> dict[str, np.ndarray]:
    branches: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        for key in sorted(data.files):
            if not (key.startswith("branch_") and key.endswith("_coords")):
                continue
            branch = key[len("branch_") : -len("_coords")]
            points = np.asarray(data[key], dtype=float)
            if len(points) > 0:
                branches[branch] = points
    return branches


def _target_point(branches: dict[str, np.ndarray], target: dict[str, Any]) -> Optional[np.ndarray]:
    branch = str(target.get("branch", ""))
    points = branches.get(branch)
    if points is None or len(points) == 0:
        return None
    index = int(target.get("index", len(points) - 1))
    index = max(0, min(index, len(points) - 1))
    return np.asarray(points[index], dtype=float)


class E2TargetReviewWindow(QMainWindow):
    BRANCH_COLORS = (
        "#f2ac32",
        "#60a5fa",
        "#7bd88f",
        "#f472b6",
        "#c084fc",
        "#22d3ee",
        "#fde047",
        "#fb7185",
    )

    def __init__(
        self,
        *,
        targets_payload: dict[str, Any],
        sample_records_by_id: dict[str, dict[str, Any]],
        review_json: Path,
    ) -> None:
        super().__init__()
        self.targets_payload = targets_payload
        self.sample_records_by_id = sample_records_by_id
        self.review_json = Path(review_json).resolve()
        self.records = list(targets_payload.get("selected_anatomies", []))
        self.index = 0
        self.review: dict[str, dict[str, Any]] = self._load_review()

        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.pv = pv
        self.setWindowTitle("E2 Target Review")
        self.resize(1500, 950)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#2A2A2A")
        splitter.addWidget(self.plotter.interactor)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        row = QHBoxLayout()
        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, max(len(self.records) - 1, 0))
        row.addWidget(self.prev_button)
        row.addWidget(self.next_button)
        row.addWidget(QLabel("Index"))
        row.addWidget(self.index_spin)
        side_layout.addLayout(row)

        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.meta_label = QLabel()
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet("color: #9aa7b5")
        side_layout.addWidget(self.title_label)
        side_layout.addWidget(self.meta_label)

        self.status_combo = QComboBox()
        self.status_combo.addItems(["unreviewed", "valid", "invalid", "unsure"])
        side_layout.addWidget(QLabel("Review status"))
        side_layout.addWidget(self.status_combo)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Notes")
        side_layout.addWidget(QLabel("Notes"))
        side_layout.addWidget(self.notes_edit)

        self.target_list = QListWidget()
        side_layout.addWidget(QLabel("Targets"))
        side_layout.addWidget(self.target_list, stretch=1)

        self.save_button = QPushButton("Save review JSON")
        side_layout.addWidget(self.save_button)

        self.prev_button.clicked.connect(lambda: self.set_index(self.index - 1))
        self.next_button.clicked.connect(lambda: self.set_index(self.index + 1))
        self.index_spin.valueChanged.connect(self.set_index)
        self.status_combo.currentTextChanged.connect(self._store_current_review)
        self.notes_edit.textChanged.connect(self._store_current_review)
        self.save_button.clicked.connect(self._save_review)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1F2328; color: #E8EDF2; }
            QPushButton, QSpinBox, QComboBox, QTextEdit, QListWidget {
                background: #11161C; color: #E8EDF2; border: 1px solid #3A4551; border-radius: 6px; padding: 5px;
            }
            QPushButton:hover { border-color: #F2AC32; }
            QLabel { color: #E8EDF2; }
            """
        )

        self.set_index(0)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Right:
            self.set_index(self.index + 1)
            return
        if event.key() == Qt.Key_Left:
            self.set_index(self.index - 1)
            return
        super().keyPressEvent(event)

    def _load_review(self) -> dict[str, dict[str, Any]]:
        if self.review_json.exists():
            try:
                payload = _load_json(self.review_json)
                if isinstance(payload, dict):
                    return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
            except Exception:
                return {}
        return {}

    def _save_review(self) -> None:
        self.review_json.parent.mkdir(parents=True, exist_ok=True)
        self.review_json.write_text(json.dumps(self.review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.statusBar().showMessage(f"Saved {self.review_json}", 4000)

    def _store_current_review(self) -> None:
        if not self.records:
            return
        record_id = str(self.records[self.index].get("record_id", ""))
        if not record_id:
            return
        status = self.status_combo.currentText()
        self.review[record_id] = {
            "record_id": record_id,
            "status": "" if status == "unreviewed" else status,
            "notes": self.notes_edit.toPlainText(),
        }

    def set_index(self, value: int) -> None:
        if not self.records:
            return
        value = max(0, min(int(value), len(self.records) - 1))
        if value != self.index:
            self._store_current_review()
        self.index = value
        if self.index_spin.value() != value:
            self.index_spin.blockSignals(True)
            self.index_spin.setValue(value)
            self.index_spin.blockSignals(False)
        self._render_record()

    def _render_record(self) -> None:
        item = dict(self.records[self.index])
        record_id = str(item["record_id"])
        sample = self.sample_records_by_id[record_id]
        mesh_path = Path(str(sample.get("visualization_mesh_path") or sample.get("simulation_mesh_path")))
        centerline_path = Path(str(sample["centerline_bundle_path"]))
        branches = _load_centerline_branches(centerline_path)

        self.plotter.clear()
        if mesh_path.exists():
            mesh = self.pv.read(str(mesh_path))
            self.plotter.add_mesh(mesh, color="#8B949E", opacity=0.28, smooth_shading=True)

        for idx, (branch_name, points) in enumerate(branches.items()):
            if len(points) < 2:
                continue
            color = "#5c7080" if branch_name.lower() == "aorta" else self.BRANCH_COLORS[idx % len(self.BRANCH_COLORS)]
            width = 3 if branch_name.lower() == "aorta" else 6
            self.plotter.add_mesh(self.pv.Spline(points, len(points)), color=color, opacity=0.85, line_width=width)

        label_points = []
        label_texts = []
        self.target_list.clear()
        for target in item.get("targets", []):
            target = dict(target)
            point = _target_point(branches, target)
            label = f"{target.get('target_index')}: {target.get('branch')} {target.get('target_position')} index={target.get('index')}"
            self.target_list.addItem(label)
            if point is None:
                continue
            self.plotter.add_mesh(self.pv.Sphere(radius=1.3, center=tuple(point)), color="#EF476F")
            label_points.append(point)
            label_texts.append(f"{target.get('branch')}:{target.get('target_position')}")
        if label_points:
            self.plotter.add_point_labels(
                np.asarray(label_points, dtype=float),
                label_texts,
                point_size=0,
                font_size=12,
                text_color="white",
                shape_color="#11161C",
                shape_opacity=0.65,
            )

        review = self.review.get(record_id, {})
        status = str(review.get("status", "")) or "unreviewed"
        self.status_combo.blockSignals(True)
        self.status_combo.setCurrentText(status)
        self.status_combo.blockSignals(False)
        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(str(review.get("notes", "")))
        self.notes_edit.blockSignals(False)

        self.title_label.setText(f"{self.index + 1}/{len(self.records)}  {record_id}")
        self.meta_label.setText(
            f"arch={item.get('arch_type')} seed={item.get('anatomy_seed')}\n"
            f"mesh={mesh_path}\ncenterline={centerline_path}\n"
            f"branches={len(branches)} targets={len(item.get('targets', []))}"
        )
        self.plotter.reset_camera()
        self.plotter.render()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    targets_payload = _load_json(args.targets_json)
    sample_records = _sample_records_by_id(targets_payload)
    app = QApplication.instance() or QApplication(sys.argv)
    window = E2TargetReviewWindow(
        targets_payload=targets_payload,
        sample_records_by_id=sample_records,
        review_json=args.review_json,
    )
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
