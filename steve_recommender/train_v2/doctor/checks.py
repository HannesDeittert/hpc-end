"""Doctor/preflight checks for train_v2."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import List

import torch

from steve_recommender.bench import resolve_device
from steve_recommender.eval_v2.discovery import FileBasedAnatomyDiscovery

from ..config import DoctorConfig
from ..runtime.intervention_factory import build_intervention
from .report import CheckResult


MONITOR_PLUGIN_PATH = (
    Path(__file__).resolve().parents[3]
    / "native"
    / "sofa_wire_force_monitor"
    / "build"
    / "libSofaWireForceMonitor.so"
)


def check_python_runtime() -> List[CheckResult]:
    """Verify the core Python runtime and imports are available."""

    required_modules = ("torch", "third_party.stEVE.eve", "third_party.stEVE_rl.eve_rl")
    results = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult("error", f"import:{module_name}", f"import failed: {exc}")
            )
        else:
            results.append(CheckResult("ok", f"import:{module_name}", "available"))
    return results


def check_sofa_runtime() -> List[CheckResult]:
    """Verify the SOFA Python runtime is discoverable."""

    results = []
    sofa_root = os.environ.get("SOFA_ROOT", "").strip()
    if not sofa_root:
        results.append(CheckResult("error", "sofa_root", "SOFA_ROOT is not set"))
    else:
        root_path = Path(sofa_root)
        if not root_path.exists():
            results.append(
                CheckResult(
                    "error", "sofa_root", f"SOFA_ROOT does not exist: {root_path}"
                )
            )
        else:
            results.append(CheckResult("ok", "sofa_root", str(root_path)))

    for module_name in ("Sofa", "SofaRuntime"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult("error", f"import:{module_name}", f"import failed: {exc}")
            )
        else:
            results.append(CheckResult("ok", f"import:{module_name}", "available"))
    return results


def check_cuda_device(*, requested_device: str) -> List[CheckResult]:
    """Verify the requested training device is usable."""

    if not requested_device.startswith("cuda"):
        return [
            CheckResult("ok", "device", f"using non-cuda device {requested_device}")
        ]
    if not torch.cuda.is_available():
        return [
            CheckResult(
                "error",
                "device",
                "CUDA requested but torch.cuda.is_available() is false",
            )
        ]
    return [CheckResult("ok", "device", f"CUDA available for {requested_device}")]


def check_anatomy_and_tool(cfg: DoctorConfig) -> List[CheckResult]:
    """Verify anatomy and wire/tool refs resolve from eval_v2 discovery."""

    results: List[CheckResult] = []
    if cfg.runtime.anatomy_id is not None:
        try:
            anatomy = FileBasedAnatomyDiscovery().get_anatomy(
                record_id=cfg.runtime.anatomy_id
            )
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult("error", "anatomy", str(exc)))
        else:
            results.append(
                CheckResult(
                    "ok",
                    "anatomy",
                    f"resolved {cfg.runtime.anatomy_id} -> {anatomy.simulation_mesh_path}",
                )
            )
    else:
        results.append(
            CheckResult(
                "ok",
                "anatomy",
                "not required for ArchVarRandom intervention path",
            )
        )

    try:
        device, resolved_label = resolve_device(
            cfg.runtime.tool_ref,
            cfg.runtime.tool_module,
            cfg.runtime.tool_class,
        )
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("error", "tool", str(exc)))
    else:
        device_name = getattr(device, "name", resolved_label)
        results.append(CheckResult("ok", "tool", f"resolved {device_name}"))
    return results


def check_resume_paths(cfg: DoctorConfig) -> List[CheckResult]:
    """Verify optional resume artifacts exist and are readable."""

    results: List[CheckResult] = []
    for label, path in (
        ("resume_from", cfg.resume_from),
        ("resume_replay_buffer_from", cfg.resume_replay_buffer_from),
    ):
        if path is None:
            continue
        if not Path(path).is_file():
            results.append(CheckResult("error", label, f"file not found: {path}"))
            continue
        try:
            torch.load(str(path), map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult("error", label, f"torch.load failed: {exc}"))
        else:
            results.append(CheckResult("ok", label, f"readable: {path}"))
    return results


def check_force_reward_prereqs(cfg: DoctorConfig) -> List[CheckResult]:
    """Verify force-reward specific prerequisites."""

    if cfg.reward.profile != "default_plus_normal_force_penalty":
        return []
    if not MONITOR_PLUGIN_PATH.exists():
        return [
            CheckResult(
                "error",
                "force_monitor_plugin",
                f"missing plugin: {MONITOR_PLUGIN_PATH}",
            )
        ]
    return [
        CheckResult("ok", "force_monitor_plugin", str(MONITOR_PLUGIN_PATH)),
    ]


def check_output_root(cfg: DoctorConfig) -> List[CheckResult]:
    """Verify the output root exists or can be created."""

    root = Path(cfg.output_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("error", "output_root", f"cannot create {root}: {exc}")]
    return [CheckResult("ok", "output_root", str(root))]


def check_env_boot(cfg: DoctorConfig) -> List[CheckResult]:
    """Build one real intervention as a preflight sanity check."""

    if not cfg.boot_env:
        return [CheckResult("ok", "env_boot", "skipped")]
    try:
        intervention = build_intervention(runtime_spec=cfg.runtime)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("error", "env_boot", f"build_intervention failed: {exc}")]

    simulation = getattr(intervention, "simulation", None)
    root = getattr(simulation, "root", None)
    if root is None:
        return [
            CheckResult(
                "warning",
                "env_boot",
                "intervention built but simulation root is not initialized yet",
            )
        ]
    return [CheckResult("ok", "env_boot", "intervention built successfully")]


def run_doctor(cfg: DoctorConfig) -> List[CheckResult]:
    """Run the full doctor/preflight set for one config."""

    results: List[CheckResult] = []
    for check in (
        check_python_runtime(),
        check_sofa_runtime(),
        check_cuda_device(requested_device=cfg.trainer_device),
        check_anatomy_and_tool(cfg),
        check_resume_paths(cfg),
        check_force_reward_prereqs(cfg),
        check_output_root(cfg),
        check_env_boot(cfg),
    ):
        results.extend(check)
    return results
