import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv
import yaml
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from steve_recommender.anatomy.aortic_arch_dataset import AorticArchRecord
from steve_recommender.evaluation.config import AorticArchSpec
from steve_recommender.rl.trained_agent_discovery import trained_checkpoints_by_tool
from steve_recommender.storage import list_models, list_wires
from steve_recommender.ui.components.anatomy_select_dialog import AnatomySelectionDialog


class _PlaybackWorker(QThread):
    frame_ready = pyqtSignal(object)
    log_line = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        tool: str,
        checkpoint: str,
        anatomy: AorticArchSpec,
        episodes: int,
        max_steps: int,
        device: str,
        base_seed: int,
        frame_stride: int,
    ) -> None:
        super().__init__()
        self.tool = tool
        self.checkpoint = checkpoint
        self.anatomy = anatomy
        self.episodes = int(episodes)
        self.max_steps = int(max_steps)
        self.device = device
        self.base_seed = int(base_seed)
        self.frame_stride = max(1, int(frame_stride))
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self.requestInterruption()

    def _should_stop(self) -> bool:
        return self._stop.is_set() or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: C901 - self-contained worker loop
        algo = None
        env = None
        try:
            import torch

            from steve_recommender.adapters import eve_rl
            from steve_recommender.evaluation.intervention_factory import (
                build_aortic_arch_intervention,
            )
            from steve_recommender.rl.bench_env import BenchEnv

            intervention, _ = build_aortic_arch_intervention(
                tool_ref=self.tool, anatomy=self.anatomy
            )
            env = BenchEnv(
                intervention=intervention,
                mode="eval",
                visualisation=True,
                n_max_steps=self.max_steps,
            )

            device = torch.device(self.device)
            algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(self.checkpoint)
            algo.to(device)

            seed = self.base_seed
            for ep in range(self.episodes):
                if self._should_stop():
                    break
                self.log_line.emit(
                    f"[ui] embedded play episode {ep + 1}/{self.episodes} seed={seed}"
                )
                algo.reset()
                obs, _ = env.reset(seed=seed)
                obs_flat, _ = eve_rl.util.flatten_obs(obs)
                step = 0

                while True:
                    if self._should_stop():
                        break
                    action = algo.get_eval_action(obs_flat)
                    if not isinstance(action, np.ndarray):
                        action = np.asarray(action, dtype=np.float32)

                    env_action = action.reshape(env.action_space.shape)
                    env_action = (env_action + 1.0) / 2.0 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low

                    obs, _, terminal, truncation, _ = env.step(env_action)
                    obs_flat, _ = eve_rl.util.flatten_obs(obs)
                    frame = env.render()
                    if frame is not None and step % self.frame_stride == 0:
                        self.frame_ready.emit(frame)

                    step += 1
                    if terminal or truncation:
                        break
                seed += 1
        except Exception as exc:  # noqa: BLE001
            self.log_line.emit(f"[ui] embedded playback error: {exc}")
        finally:
            if algo is not None:
                try:
                    algo.close()
                except Exception:
                    pass
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self.finished.emit()


class EvaluateWidget(QWidget):
    """UI to create an evaluation config and launch the evaluator."""

    def __init__(self):
        super().__init__()

        self._selected_anatomy: Optional[AorticArchRecord] = None
        self._selected_centerline_npz: Optional[Path] = None
        self._trained_by_tool = trained_checkpoints_by_tool()

        # --- Anatomy selection ---
        self.btn_select_anatomy = QPushButton("Select Anatomy…")
        self.btn_select_anatomy.clicked.connect(self._select_anatomy)
        self.lbl_anatomy = QLabel("No anatomy selected.")
        self.lbl_anatomy.setWordWrap(True)

        self.preview = QtInteractor(self)
        self.preview.setMinimumHeight(250)

        # --- Agent table ---
        self.agent_table = QTableWidget(0, 4)
        self.agent_table.setHorizontalHeaderLabels(["Agent name", "Tool", "Checkpoint (.everl)", ""])
        self.agent_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.agent_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.agent_table.setColumnWidth(3, 80)

        self.btn_add_agent = QPushButton("Add agent")
        self.btn_add_agent.clicked.connect(self._add_agent_row)

        self.chk_only_trained_tools = QCheckBox("Only show tools with checkpoints")
        self.chk_only_trained_tools.setChecked(bool(self._trained_by_tool))
        self.btn_refresh_trained = QPushButton("Refresh checkpoints")
        self.btn_refresh_trained.clicked.connect(self._refresh_trained_checkpoints)

        # --- Evaluation params ---
        self.run_name = QLineEdit("ui_eval")
        self.n_trials = QSpinBox()
        self.n_trials.setRange(1, 100000)
        self.n_trials.setValue(10)
        self.base_seed = QSpinBox()
        self.base_seed.setRange(0, 2**31 - 1)
        self.base_seed.setValue(123)
        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 10_000_000)
        self.max_steps.setValue(1000)

        self.policy_device = QComboBox()
        self.policy_device.addItems(["cuda", "cpu"])
        self.use_non_mp = QCheckBox("Use non-mp SOFA (needed for force proxies)")
        self.use_non_mp.setChecked(True)

        self.target_branches = QLineEdit("lcca")
        self.target_threshold = QDoubleSpinBox()
        self.target_threshold.setRange(0.1, 1000.0)
        self.target_threshold.setValue(5.0)

        # --- Scoring (simple controls; advanced users can edit YAML later) ---
        self.score_force_scale = QDoubleSpinBox()
        self.score_force_scale.setRange(1e-6, 1e9)
        self.score_force_scale.setValue(1.0)
        self.score_lcp_scale = QDoubleSpinBox()
        self.score_lcp_scale.setRange(1e-6, 1e9)
        self.score_lcp_scale.setValue(1.0)
        self.score_speed_scale = QDoubleSpinBox()
        self.score_speed_scale.setRange(1e-6, 1e9)
        self.score_speed_scale.setValue(50.0)

        # --- Run controls ---
        self.btn_write_config = QPushButton("Write config")
        self.btn_write_config.clicked.connect(self._write_config_only)
        self.btn_run = QPushButton("Start evaluation")
        self.btn_run.clicked.connect(self._start_eval)
        self.btn_run.setEnabled(False)
        self.btn_play = QPushButton("Play first agent (Sofa window)")
        self.btn_play.clicked.connect(self._start_play)
        self.btn_play.setEnabled(False)

        # --- Embedded playback ---
        self.play_episodes = QSpinBox()
        self.play_episodes.setRange(1, 1000)
        self.play_episodes.setValue(1)
        self.play_frame_stride = QSpinBox()
        self.play_frame_stride.setRange(1, 120)
        self.play_frame_stride.setValue(2)

        self.btn_play_embedded = QPushButton("Play embedded")
        self.btn_play_embedded.clicked.connect(self._start_play_embedded)
        self.btn_play_embedded.setEnabled(False)
        self.btn_stop_embedded = QPushButton("Stop playback")
        self.btn_stop_embedded.clicked.connect(self._stop_play_embedded)
        self.btn_stop_embedded.setEnabled(False)

        self.playback_view = QLabel("Embedded playback will appear here.")
        self.playback_view.setAlignment(Qt.AlignCenter)
        self.playback_view.setMinimumHeight(240)
        self.playback_view.setStyleSheet(
            "background: #101214; color: #d8dbe2; border: 1px solid #2b2f36;"
        )

        self.log = QTextEdit(readOnly=True)
        self.log.setPlaceholderText("Evaluation output will appear here…")

        self._playback_worker: Optional[_PlaybackWorker] = None
        self._last_pixmap: Optional[QPixmap] = None

        # Layout
        anatomy_box = QGroupBox("1) Anatomy")
        anatomy_layout = QVBoxLayout(anatomy_box)
        anatomy_top = QHBoxLayout()
        anatomy_top.addWidget(self.btn_select_anatomy)
        anatomy_top.addWidget(self.lbl_anatomy, 1)
        anatomy_layout.addLayout(anatomy_top)
        anatomy_layout.addWidget(self.preview, 1)

        agents_box = QGroupBox("2) Agents (tool + checkpoint)")
        agents_layout = QVBoxLayout(agents_box)
        agents_layout.addWidget(self.agent_table, 1)
        agents_buttons = QHBoxLayout()
        agents_buttons.addWidget(self.btn_add_agent)
        agents_buttons.addStretch()
        agents_buttons.addWidget(self.chk_only_trained_tools)
        agents_buttons.addWidget(self.btn_refresh_trained)
        agents_layout.addLayout(agents_buttons)

        params_box = QGroupBox("3) Parameters")
        form = QFormLayout(params_box)
        form.addRow("Run name", self.run_name)
        form.addRow("Trials per agent", self.n_trials)
        form.addRow("Base seed", self.base_seed)
        form.addRow("Max episode steps", self.max_steps)
        form.addRow("Policy device", self.policy_device)
        form.addRow("", self.use_non_mp)
        form.addRow("Target branches (comma)", self.target_branches)
        form.addRow("Target threshold (mm)", self.target_threshold)

        scoring_box = QGroupBox("4) Scoring (default_v1 scales)")
        scoring_form = QFormLayout(scoring_box)
        scoring_form.addRow("force_scale", self.score_force_scale)
        scoring_form.addRow("lcp_scale", self.score_lcp_scale)
        scoring_form.addRow("speed_scale_mm_s", self.score_speed_scale)

        run_box = QGroupBox("5) Run")
        run_layout = QHBoxLayout(run_box)
        run_layout.addWidget(self.btn_write_config)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_play)
        run_layout.addStretch()

        playback_box = QGroupBox("6) Playback (embedded)")
        playback_layout = QVBoxLayout(playback_box)
        playback_layout.addWidget(self.playback_view, 1)
        playback_controls = QHBoxLayout()
        playback_controls.addWidget(QLabel("Episodes"))
        playback_controls.addWidget(self.play_episodes)
        playback_controls.addWidget(QLabel("Render stride"))
        playback_controls.addWidget(self.play_frame_stride)
        playback_controls.addStretch()
        playback_controls.addWidget(self.btn_play_embedded)
        playback_controls.addWidget(self.btn_stop_embedded)
        playback_layout.addLayout(playback_controls)

        left = QVBoxLayout()
        left.addWidget(anatomy_box, 2)
        left.addWidget(agents_box, 2)

        right = QVBoxLayout()
        right.addWidget(params_box, 0)
        right.addWidget(scoring_box, 0)
        right.addWidget(run_box, 0)
        right.addWidget(playback_box, 2)
        right.addWidget(self.log, 1)

        main = QHBoxLayout(self)
        main.addLayout(left, 3)
        main.addLayout(right, 2)

        # Start with one empty agent row.
        self._add_agent_row()

    # ------------------------------------------------------------------
    # Anatomy selection + preview
    # ------------------------------------------------------------------
    def _select_anatomy(self):
        dlg = AnatomySelectionDialog(self)
        if dlg.exec_() != dlg.Accepted or dlg.selected_record is None:
            return
        self._selected_anatomy = dlg.selected_record

        # Compute absolute path to centerline.npz (if available)
        self._selected_centerline_npz = dlg.resolve_centerline_path(dlg.selected_record)

        self.lbl_anatomy.setText(
            f"{dlg.selected_record.record_id} | arch_type={dlg.selected_record.arch_type} seed={dlg.selected_record.seed}"
        )
        self._render_centerlines()
        self._update_run_enabled()

    def _render_centerlines(self):
        self.preview.clear()
        if not self._selected_centerline_npz or not self._selected_centerline_npz.exists():
            self.preview.add_text("No centerline data.", font_size=12)
            return
        data = np.load(self._selected_centerline_npz, allow_pickle=True)
        branch_names = data.get("branch_names", [])
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        for i, name in enumerate(branch_names.tolist() if hasattr(branch_names, "tolist") else []):
            coords_key = f"branch_{name}_coords"
            if coords_key not in data:
                continue
            pts = np.asarray(data[coords_key], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])]).astype(np.int64)
            poly = pv.PolyData(pts, lines)
            self.preview.add_mesh(poly, color=colors[i % len(colors)], line_width=4)
        self.preview.reset_camera()

    # ------------------------------------------------------------------
    # Agents selection
    # ------------------------------------------------------------------
    def _tool_options(self) -> List[str]:
        if self.chk_only_trained_tools.isChecked() and self._trained_by_tool:
            return sorted(self._trained_by_tool.keys())

        tools: List[str] = []
        for model in list_models():
            for wire in list_wires(model):
                tools.append(f"{model}/{wire}")
        return tools

    def _add_agent_row(self):
        row = self.agent_table.rowCount()
        self.agent_table.insertRow(row)

        name_item = QTableWidgetItem(f"agent_{row}")
        self.agent_table.setItem(row, 0, name_item)

        tool_box = QComboBox()
        tool_box.addItems(self._tool_options())
        self.agent_table.setCellWidget(row, 1, tool_box)

        checkpoint_edit = QLineEdit()
        checkpoint_edit.setPlaceholderText("Select a checkpoint .everl file…")
        self.agent_table.setCellWidget(row, 2, checkpoint_edit)

        btn = QPushButton("Browse")
        btn.clicked.connect(lambda _=None, r=row: self._browse_checkpoint(r))
        self.agent_table.setCellWidget(row, 3, btn)

        tool_box.currentTextChanged.connect(lambda _: self._maybe_autofill_checkpoint(row))
        checkpoint_edit.textChanged.connect(lambda _: self._update_run_enabled())
        self._maybe_autofill_checkpoint(row)

    def _refresh_trained_checkpoints(self):
        self._trained_by_tool = trained_checkpoints_by_tool()
        if not self._trained_by_tool:
            self.chk_only_trained_tools.setChecked(False)
        self._rebuild_tool_dropdowns()

    def _rebuild_tool_dropdowns(self):
        options = self._tool_options()
        for row in range(self.agent_table.rowCount()):
            tool_box: QComboBox = self.agent_table.cellWidget(row, 1)  # type: ignore[assignment]
            if tool_box is None:
                continue
            current = tool_box.currentText()
            tool_box.blockSignals(True)
            tool_box.clear()
            tool_box.addItems(options)
            if current and current in options:
                tool_box.setCurrentText(current)
            tool_box.blockSignals(False)
            self._maybe_autofill_checkpoint(row)

    def _maybe_autofill_checkpoint(self, row: int):
        tool_box: QComboBox = self.agent_table.cellWidget(row, 1)  # type: ignore[assignment]
        ckpt_edit: QLineEdit = self.agent_table.cellWidget(row, 2)  # type: ignore[assignment]
        if tool_box is None or ckpt_edit is None:
            return
        if ckpt_edit.text().strip():
            return

        tool = tool_box.currentText().strip()
        candidates = self._trained_by_tool.get(tool) if tool else None
        if not candidates:
            return
        ckpt_edit.setText(str(candidates[0].checkpoint_path))

    def _browse_checkpoint(self, row: int):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            str(Path.cwd() / "results"),
            "EveRL checkpoints (*.everl);;All files (*)",
        )
        if not path:
            return
        edit: QLineEdit = self.agent_table.cellWidget(row, 2)  # type: ignore[assignment]
        edit.setText(path)
        self._update_run_enabled()

    def _collect_agents(self) -> List[Dict[str, str]]:
        agents: List[Dict[str, str]] = []
        for row in range(self.agent_table.rowCount()):
            name = self.agent_table.item(row, 0).text() if self.agent_table.item(row, 0) else f"agent_{row}"
            tool_box: QComboBox = self.agent_table.cellWidget(row, 1)  # type: ignore[assignment]
            tool = tool_box.currentText() if tool_box else ""
            ckpt_edit: QLineEdit = self.agent_table.cellWidget(row, 2)  # type: ignore[assignment]
            ckpt = ckpt_edit.text().strip() if ckpt_edit else ""
            if not tool or not ckpt:
                continue
            agents.append({"name": name, "tool": tool, "checkpoint": ckpt})
        return agents

    # ------------------------------------------------------------------
    # Config writing + evaluation process
    # ------------------------------------------------------------------
    def _build_anatomy_spec(self) -> AorticArchSpec:
        if self._selected_anatomy is None:
            raise RuntimeError("No anatomy selected")

        branches = [b.strip() for b in (self.target_branches.text() or "").split(",") if b.strip()]
        if not branches:
            branches = ["lcca"]

        return AorticArchSpec(
            arch_type=self._selected_anatomy.arch_type,
            seed=int(self._selected_anatomy.seed),
            rotation_yzx_deg=list(self._selected_anatomy.rotation_yzx_deg)
            if self._selected_anatomy.rotation_yzx_deg
            else None,
            scaling_xyzd=list(self._selected_anatomy.scaling_xyzd)
            if self._selected_anatomy.scaling_xyzd
            else None,
            omit_axis=self._selected_anatomy.omit_axis,
            target_branches=branches,
            target_threshold_mm=float(self.target_threshold.value()),
        )

    def _build_config_dict(self) -> Dict:
        anatomy_spec = self._build_anatomy_spec()
        anatomy = {
            "type": "aortic_arch",
            "arch_type": anatomy_spec.arch_type,
            "seed": int(anatomy_spec.seed),
            "rotation_yzx_deg": list(anatomy_spec.rotation_yzx_deg) if anatomy_spec.rotation_yzx_deg else None,
            "scaling_xyzd": list(anatomy_spec.scaling_xyzd) if anatomy_spec.scaling_xyzd else None,
            "omit_axis": anatomy_spec.omit_axis,
            "target_mode": anatomy_spec.target_mode,
            "target_branches": list(anatomy_spec.target_branches),
            "target_threshold_mm": float(anatomy_spec.target_threshold_mm),
        }

        cfg = {
            "name": self.run_name.text().strip() or "ui_eval",
            "agents": self._collect_agents(),
            "n_trials": int(self.n_trials.value()),
            "base_seed": int(self.base_seed.value()),
            "max_episode_steps": int(self.max_steps.value()),
            "policy_device": self.policy_device.currentText(),
            "use_non_mp_sim": bool(self.use_non_mp.isChecked()),
            "output_root": "results/eval_runs",
            "anatomy": anatomy,
            "scoring": {
                "mode": "default_v1",
                "force_scale": float(self.score_force_scale.value()),
                "lcp_scale": float(self.score_lcp_scale.value()),
                "speed_scale_mm_s": float(self.score_speed_scale.value()),
            },
        }
        return cfg

    def _write_config_only(self):
        try:
            cfg = self._build_config_dict()
        except Exception as e:
            self.log.append(f"[ui] error: {e}")
            return

        cfg_dir = Path("results/eval_configs")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        path = cfg_dir / f"{ts}_{cfg['name']}.yml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self.log.append(f"[ui] wrote config: {path}")

    def _start_eval(self):
        try:
            cfg = self._build_config_dict()
        except Exception as e:
            self.log.append(f"[ui] error: {e}")
            return

        cfg_dir = Path("results/eval_configs")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        cfg_path = cfg_dir / f"{ts}_{cfg['name']}.yml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self.log.append(f"[ui] starting evaluation with config: {cfg_path}")

        cmd = [
            sys.executable,
            "-m",
            "steve_recommender.evaluation.run_cli",
            "--config",
            str(cfg_path),
        ]

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessEnvironment(self._build_sofa_env())
        proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(proc))
        proc.readyReadStandardError.connect(lambda: self._append_proc_error(proc))
        proc.finished.connect(lambda code, status: self.log.append(f"[ui] finished: code={code} status={status}"))
        proc.start()

        self._proc = proc  # keep reference

    def _append_proc_output(self, proc: QProcess):
        out = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if out:
            self.log.append(out.rstrip())

    def _append_proc_error(self, proc: QProcess):
        out = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")
        if out:
            self.log.append(out.rstrip())

    def _sofa_env_overrides(self) -> Dict[str, str]:
        overrides: Dict[str, str] = {}

        # Detect conda prefix (used for libpython3.8.so resolution in SofaPython3).
        conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix

        sofa_root = os.environ.get("SOFA_ROOT")
        if not sofa_root:
            candidates = [
                str(Path.home() / "opt" / "Sofa_v23.06.00_Linux" / "SOFA_v23.06.00_Linux"),
            ]
            for c in candidates:
                if Path(c, "plugins", "SofaPython3").exists():
                    sofa_root = c
                    break

        if sofa_root:
            overrides["SOFA_ROOT"] = sofa_root
            sofa_py = str(
                Path(sofa_root) / "plugins" / "SofaPython3" / "lib" / "python3" / "site-packages"
            )
            old_py = os.environ.get("PYTHONPATH", "")
            if sofa_py in old_py.split(":"):
                overrides["PYTHONPATH"] = old_py
            else:
                overrides["PYTHONPATH"] = f"{sofa_py}:{old_py}" if old_py else sofa_py

        # Prepend conda lib path
        conda_lib = str(Path(conda_prefix) / "lib")
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if conda_lib in old_ld.split(":"):
            overrides["LD_LIBRARY_PATH"] = old_ld
        else:
            overrides["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else conda_lib

        return overrides

    def _apply_sofa_env_in_process(self) -> None:
        for key, value in self._sofa_env_overrides().items():
            os.environ[key] = value

    def _build_sofa_env(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        for key, value in self._sofa_env_overrides().items():
            env.insert(key, value)
        return env

    def _is_playback_running(self) -> bool:
        return self._playback_worker is not None and self._playback_worker.isRunning()

    def _update_run_enabled(self):
        has_agents = len(self._collect_agents()) > 0
        has_anatomy = self._selected_anatomy is not None
        enabled = bool(has_agents and has_anatomy)
        self.btn_run.setEnabled(enabled)
        self.btn_play.setEnabled(enabled)
        if self._is_playback_running():
            self.btn_play_embedded.setEnabled(False)
            self.btn_stop_embedded.setEnabled(True)
        else:
            self.btn_play_embedded.setEnabled(enabled)
            self.btn_stop_embedded.setEnabled(False)

    # ------------------------------------------------------------------
    # Embedded playback
    # ------------------------------------------------------------------
    def _start_play_embedded(self):
        if self._is_playback_running():
            self.log.append("[ui] embedded playback already running")
            return

        if self._selected_anatomy is None:
            self.log.append("[ui] no anatomy selected")
            return

        agents = self._collect_agents()
        if not agents:
            self.log.append("[ui] no agent configured")
            return

        try:
            anatomy_spec = self._build_anatomy_spec()
        except Exception as e:
            self.log.append(f"[ui] error: {e}")
            return

        agent = agents[0]
        tool = agent["tool"]
        checkpoint = agent["checkpoint"]

        self._apply_sofa_env_in_process()
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "-2000,-2000")

        self.playback_view.clear()
        self.playback_view.setText("Starting embedded playback...")
        self._last_pixmap = None

        worker = _PlaybackWorker(
            tool=tool,
            checkpoint=checkpoint,
            anatomy=anatomy_spec,
            episodes=int(self.play_episodes.value()),
            max_steps=int(self.max_steps.value()),
            device=self.policy_device.currentText(),
            base_seed=int(self.base_seed.value()),
            frame_stride=int(self.play_frame_stride.value()),
        )
        worker.frame_ready.connect(self._update_playback_frame)
        worker.log_line.connect(lambda line: self.log.append(line))
        worker.finished.connect(self._on_playback_finished)
        self._playback_worker = worker
        worker.start()
        self._update_run_enabled()

        self.log.append(
            f"[ui] embedded playback started: tool={tool} ckpt={checkpoint} arch={self._selected_anatomy.record_id}"
        )

    def _stop_play_embedded(self):
        if not self._is_playback_running():
            return
        self.log.append("[ui] stopping embedded playback...")
        if self._playback_worker is not None:
            self._playback_worker.stop()

    def _on_playback_finished(self):
        self._playback_worker = None
        self._update_run_enabled()
        self.log.append("[ui] embedded playback finished")

    def _update_playback_frame(self, frame: np.ndarray):
        if frame is None:
            return
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        height, width, channels = frame.shape
        fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
        image = QImage(frame.data, width, height, frame.strides[0], fmt).copy()
        self._last_pixmap = QPixmap.fromImage(image)
        self._set_playback_pixmap()

    def _set_playback_pixmap(self) -> None:
        if self._last_pixmap is None:
            return
        target = self.playback_view.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._last_pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.playback_view.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Play agent on anatomy (SofaPygame)
    # ------------------------------------------------------------------
    def _start_play(self):
        if self._selected_anatomy is None:
            self.log.append("[ui] no anatomy selected")
            return

        agents = self._collect_agents()
        if not agents:
            self.log.append("[ui] no agent configured")
            return

        # For now, use the first configured agent row.
        agent = agents[0]
        tool = agent["tool"]
        checkpoint = agent["checkpoint"]

        cfg_dir = Path("results/eval_configs")
        cfg_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "steve_recommender.evaluation.play_agent",
            "--tool",
            tool,
            "--checkpoint",
            checkpoint,
            "--arch-record",
            self._selected_anatomy.record_id,
        ]

        self.log.append(f"[ui] starting play: tool={tool} ckpt={checkpoint} arch={self._selected_anatomy.record_id}")

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setProcessEnvironment(self._build_sofa_env())
        proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(proc))
        proc.readyReadStandardError.connect(lambda: self._append_proc_error(proc))
        proc.finished.connect(lambda code, status: self.log.append(f"[ui] play finished: code={code} status={status}"))
        proc.start()

        self._play_proc = proc  # keep reference

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming convention
        super().resizeEvent(event)
        self._set_playback_pixmap()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming convention
        self._stop_play_embedded()
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._playback_worker.wait(2000)
        super().closeEvent(event)
