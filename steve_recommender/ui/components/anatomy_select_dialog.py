from pathlib import Path
from typing import List, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from steve_recommender.anatomy.aortic_arch_dataset import (
    AorticArchRecord,
    load_aortic_arch_dataset,
)


class _AnatomyTableModel(QAbstractTableModel):
    COLUMNS = ("id", "arch_type", "seed", "created_at")

    def __init__(self, records: List[AorticArchRecord]) -> None:
        super().__init__()
        self._all = list(records)
        self._view = list(records)
        self._query = ""
        self._arch_type = "All"

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._view)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self.COLUMNS):
            return self.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._view)):
            return None
        record = self._view[index.row()]
        col = self.COLUMNS[index.column()]
        if role == Qt.DisplayRole:
            if col == "id":
                return record.record_id
            if col == "arch_type":
                return record.arch_type
            if col == "seed":
                return str(record.seed)
            if col == "created_at":
                return record.created_at
        return None

    def record_at(self, row: int) -> Optional[AorticArchRecord]:
        if 0 <= row < len(self._view):
            return self._view[row]
        return None

    def set_filters(self, *, query: str, arch_type: str) -> None:
        query = (query or "").strip().lower()
        arch_type = arch_type or "All"
        if query == self._query and arch_type == self._arch_type:
            return
        self._query = query
        self._arch_type = arch_type
        self.beginResetModel()
        self._view = []
        for r in self._all:
            if arch_type != "All" and r.arch_type != arch_type:
                continue
            if query and query not in r.record_id.lower():
                continue
            self._view.append(r)
        self.endResetModel()


class AnatomySelectionDialog(QDialog):
    """Select a stored aortic-arch anatomy from `results/anatomies/...`.

    This dialog is optimized for large sets (e.g. 10k records) by using a table model.
    """

    def __init__(self, parent: QWidget, dataset_root: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Anatomy (AorticArch dataset)")
        self.setMinimumSize(900, 500)

        self.selected_record: Optional[AorticArchRecord] = None

        self._dataset = load_aortic_arch_dataset(dataset_root)
        records = list(self._dataset.iter_index())

        # Top filter row
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by id (e.g. arch_000123)…")
        self.filter_arch = QComboBox()
        self.filter_arch.addItem("All")
        for t in sorted({r.arch_type for r in records}):
            self.filter_arch.addItem(t)

        # Table + model
        self.model = _AnatomyTableModel(records)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.doubleClicked.connect(lambda _: self._accept_if_selected())
        self.table.horizontalHeader().setStretchLastSection(True)

        self.search.textChanged.connect(self._apply_filters)
        self.filter_arch.currentTextChanged.connect(lambda _: self._apply_filters())
        self._apply_filters()

        # Preview (centerlines)
        self.preview = QtInteractor(self)
        self.preview.setMinimumWidth(400)

        # Details label
        self.details = QLabel("Select a row to preview.")
        self.details.setWordWrap(True)

        # Buttons
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        self.btn_select = QPushButton("Select")
        self.btn_select.clicked.connect(self._accept_if_selected)
        self.btn_select.setEnabled(False)

        # Layout
        left = QVBoxLayout()
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.search, 2)
        filter_row.addWidget(self.filter_arch, 1)
        left.addLayout(filter_row)
        left.addWidget(self.table, 1)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_select)
        left.addLayout(btn_row)

        right = QVBoxLayout()
        right.addWidget(self.details, 0)
        right.addWidget(self.preview, 1)

        main = QHBoxLayout(self)
        main.addLayout(left, 2)
        main.addLayout(right, 3)

        # React to selection changes
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

    @property
    def dataset_root(self) -> Path:
        return self._dataset.root

    def resolve_centerline_path(self, record: AorticArchRecord) -> Optional[Path]:
        if not record.centerline_npz:
            return None
        return self._dataset.root / record.centerline_npz

    def _apply_filters(self) -> None:
        self.model.set_filters(query=self.search.text(), arch_type=self.filter_arch.currentText())

    def _on_selection_changed(self) -> None:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            self.selected_record = None
            self.btn_select.setEnabled(False)
            self.details.setText("Select a row to preview.")
            self.preview.clear()
            return

        record = self.model.record_at(idxs[0].row())
        self.selected_record = record
        self.btn_select.setEnabled(record is not None)
        if record is None:
            return

        # Update details + preview
        self.details.setText(
            f"id={record.record_id}\narch_type={record.arch_type} seed={record.seed}\n"
            f"rotation_yzx_deg={record.rotation_yzx_deg}\nscaling_xyzd={record.scaling_xyzd}"
        )
        self._show_centerlines(record)

    def _show_centerlines(self, record: AorticArchRecord) -> None:
        self.preview.clear()

        if not record.centerline_npz:
            self.preview.add_text("No centerline data available.", font_size=12)
            return

        npz_path = self.resolve_centerline_path(record)
        if npz_path is None:
            self.preview.add_text("No centerline data available.", font_size=12)
            return
        if not npz_path.exists():
            self.preview.add_text(f"Missing: {npz_path}", font_size=12)
            return

        data = np.load(npz_path, allow_pickle=True)
        branch_names = data.get("branch_names", None)
        if branch_names is None:
            self.preview.add_text("centerline.npz missing branch_names", font_size=12)
            return

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        for i, name in enumerate(branch_names.tolist()):
            coords_key = f"branch_{name}_coords"
            if coords_key not in data:
                continue
            pts = np.asarray(data[coords_key], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            # Build a polyline cell: [n_points, 0,1,2,...]
            lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])]).astype(np.int64)
            poly = pv.PolyData(pts, lines)
            self.preview.add_mesh(poly, color=colors[i % len(colors)], line_width=4)

        self.preview.reset_camera()

    def _accept_if_selected(self) -> None:
        if self.selected_record is None:
            return
        self.accept()
