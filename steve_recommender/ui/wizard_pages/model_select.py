from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLineEdit,
    QWizardPage,
    QWidget,
)

from steve_recommender.storage import ensure_model, list_models


class ModelSelectPage(QWizardPage):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Model")
        self.setSubTitle("Select an existing model or create a new one.")

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_change)

        self.new_name = QLineEdit()
        self.new_desc = QLineEdit()

        form = QFormLayout()
        form.addRow("Model:", self.model_combo)
        form.addRow("New model name:", self.new_name)
        form.addRow("New model description:", self.new_desc)
        self.setLayout(form)

        self._reload()
        self._on_change()

    def initializePage(self) -> None:
        self._reload()
        self._on_change()

    def _reload(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("<Create new model…>")
        for name in list_models():
            self.model_combo.addItem(name)
        if current:
            idx = self.model_combo.findText(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _on_change(self) -> None:
        creating_new = self.model_combo.currentIndex() == 0
        self.new_name.setEnabled(creating_new)
        self.new_desc.setEnabled(creating_new)

    def validatePage(self) -> bool:
        if self.model_combo.currentIndex() == 0:
            name = (self.new_name.text() or "").strip()
            if not name:
                return False
            desc = (self.new_desc.text() or "").strip()
            ensure_model(name, description=desc)
            self.wizard().model_name = name
            return True

        self.wizard().model_name = self.model_combo.currentText()
        return True

