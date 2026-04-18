from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from steve_recommender.adapters import eve
from steve_recommender.devices import make_device

from .config import ForceUnitsConfig
from .info_collectors import SofaWallForceInfo


VALIDATED_ASSOC_METHODS = {
    "native_contact_export_triangle_id",
    "native_contact_export_triangle_id_global_row_bridge",
}

POINT_REFERENCE_Y_OFFSET_MM = 5.0
LINE_REFERENCE_Y_OFFSET_MM = 0.08


@dataclass(frozen=True)
class ReferenceCaseConfig:
    name: str
    expected_contact_mode: str
    require_active_constraint: bool
    steps: int
    action_insert: float
    action_rotate: float
    seed: int
    enable_intrusive_lcp: bool
    line_offset_z: float


@dataclass(frozen=True)
class ReferenceCaseSummary:
    case_name: str
    expected_contact_mode: str
    require_active_constraint: bool
    step_count: int
    active_constraint_steps: int
    point_contacts_observed: bool
    line_contacts_observed: bool
    native_records_observed: bool
    full_explicit_coverage_on_active_steps: bool
    strict_validated_on_active_steps: bool
    pass_validated_case: bool
    dominant_failure_reason: str
    failure_reasons: List[str]


@dataclass(frozen=True)
class ReferenceCaseReport:
    config: ReferenceCaseConfig
    run_a: ReferenceCaseSummary
    run_b: ReferenceCaseSummary
    reproducible: bool
    reproducibility_error: str
    first_divergence_step: int
    first_divergence_domain: str
    first_divergence_kind: str
    first_divergence_detail: str
    first_nonsemantic_divergence: str
    pass_validated_case: bool


@dataclass(frozen=True)
class ReferenceSuiteReport:
    created_at: str
    tool_ref: str
    output_dir: str
    pass_validated_suite: bool
    external_limit_detected: bool
    external_limit_reason: str
    case_reports: List[ReferenceCaseReport]


def _write_planar_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "v -100 -40 0",
            "v  120 -40 0",
            "v  120  40 0",
            "v -100  40 0",
            "f 1 2 3",
            "f 1 3 4",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _write_tube_mesh(
    path: Path,
    *,
    radius_mm: float,
    length_mm: float,
    segments: int,
    rings: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts: List[Tuple[float, float, float]] = []
    for ring in range(rings + 1):
        x = float(ring) * float(length_mm) / float(rings)
        for seg in range(segments):
            angle = (2.0 * math.pi * float(seg)) / float(segments)
            y = float(radius_mm) * math.cos(angle)
            z = float(radius_mm) * math.sin(angle)
            verts.append((x, y, z))

    faces: List[Tuple[int, int, int]] = []
    for ring in range(rings):
        base = ring * segments
        nxt = (ring + 1) * segments
        for seg in range(segments):
            seg_next = (seg + 1) % segments
            a = base + seg
            b = base + seg_next
            c = nxt + seg
            d = nxt + seg_next
            faces.append((a, b, d))
            faces.append((a, d, c))

    with path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _collision_contact_counts(intervention: Any) -> Tuple[int, int]:
    sim = getattr(intervention, "simulation", None)
    if sim is None:
        return 0, 0
    try:
        point_contacts = int(
            sim._instruments_combined.CollisionModel.PointCollisionModel.numberOfContacts.value  # noqa: SLF001
        )
    except Exception:
        point_contacts = 0
    try:
        line_contacts = int(
            sim._instruments_combined.CollisionModel.LineCollisionModel.numberOfContacts.value  # noqa: SLF001
        )
    except Exception:
        line_contacts = 0
    return point_contacts, line_contacts


def _build_case_intervention(
    *,
    case: ReferenceCaseConfig,
    tool_ref: str,
    mesh_path: Path,
) -> Any:
    if case.name == "point_vs_triangle":
        z = float(max(case.line_offset_z, 0.0))
        centerline = np.stack(
            [
                np.linspace(-80.0, 100.0, 200, dtype=np.float32),
                np.full((200,), float(POINT_REFERENCE_Y_OFFSET_MM), dtype=np.float32),
                np.full((200,), z, dtype=np.float32),
            ],
            axis=1,
        )
        insertion = (-80.0, float(POINT_REFERENCE_Y_OFFSET_MM), z)
        approx_radius = 1.0
    elif case.name == "line_vs_triangle":
        z = float(case.line_offset_z)
        centerline = np.stack(
            [
                np.linspace(0.0, 40.0, 160, dtype=np.float32),
                np.full((160,), float(LINE_REFERENCE_Y_OFFSET_MM), dtype=np.float32),
                np.full((160,), z, dtype=np.float32),
            ],
            axis=1,
        )
        insertion = (0.0, float(LINE_REFERENCE_Y_OFFSET_MM), z)
        approx_radius = 0.6
    else:
        raise ValueError(f"Unsupported reference case: {case.name}")

    branch = eve.intervention.vesseltree.Branch(name="main", coordinates=centerline)
    vessel_tree = eve.intervention.vesseltree.FromMesh(
        mesh=str(mesh_path),
        insertion_position=insertion,
        insertion_direction=(1.0, 0.0, 0.0),
        branch_list=[branch],
        approx_branch_radii=approx_radius,
        rotation_yzx_deg=(0.0, 0.0, 0.0),
    )

    device = make_device(tool_ref)
    simulation = eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter(friction=0.001)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
        image_rot_zx=[0, 0],
    )
    target = eve.intervention.target.BranchEnd(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=3.0,
        branches=["main"],
    )

    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=False,
        normalize_action=True,
    )
    intervention.make_non_mp()
    return intervention


def _strict_step_validated(step: Dict[str, Any]) -> bool:
    if not bool(step.get("active_constraint_step", False)):
        return True
    if str(step.get("quality_tier", "")) != "validated":
        return False
    if str(step.get("association_method", "")) not in VALIDATED_ASSOC_METHODS:
        return False
    coverage = float(step.get("association_coverage", float("nan")))
    explicit_cov = float(step.get("association_explicit_force_coverage", float("nan")))
    if not np.isfinite(coverage) or coverage < (1.0 - 1e-6):
        return False
    if not np.isfinite(explicit_cov) or explicit_cov < (1.0 - 1e-6):
        return False
    if int(step.get("gap_unmapped_count", 0)) != 0:
        return False
    if not bool(step.get("association_ordering_stable", False)):
        return False
    return True


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _quantize_xyz(vec: Any, scale: float = 1e6) -> Tuple[int, int, int]:
    arr = np.asarray(vec, dtype=np.float64).reshape((3,))
    return (
        int(np.round(arr[0] * scale)),
        int(np.round(arr[1] * scale)),
        int(np.round(arr[2] * scale)),
    )


def _semantic_step_signature(step: Dict[str, Any]) -> Tuple[Any, ...]:
    active = bool(step.get("active_constraint_step", False))
    gap_active = int(step.get("gap_active_projected_count", 0))
    gap_explicit = int(step.get("gap_explicit_mapped_count", 0))
    gap_unmapped = int(step.get("gap_unmapped_count", 0))
    full_explicit_coverage = bool(gap_active == gap_explicit and gap_unmapped == 0)
    contact_mode = str(step.get("gap_contact_mode", "none") or "none")
    has_point_mode = contact_mode in {"point", "mixed"} or bool(int(step.get("point_contacts", 0)) > 0)
    has_line_mode = contact_mode in {"line", "mixed"} or bool(int(step.get("line_contacts", 0)) > 0)
    return (
        str(step.get("quality_tier", "")),
        str(step.get("association_method", "")),
        bool(active),
        bool(full_explicit_coverage),
        bool(has_point_mode),
        bool(has_line_mode),
        bool(step.get("native_available", False)),
        bool(step.get("association_ordering_stable", False)),
    )


def _summarize_case(case: ReferenceCaseConfig, steps: Sequence[Dict[str, Any]]) -> ReferenceCaseSummary:
    active_steps = [s for s in steps if bool(s.get("active_constraint_step", False))]
    point_contacts_observed = any(
        str(s.get("gap_contact_mode", "none")) in {"point", "mixed"}
        or int(s.get("point_contacts", 0)) > 0
        for s in steps
    )
    line_contacts_observed = any(
        str(s.get("gap_contact_mode", "none")) in {"line", "mixed"}
        or int(s.get("line_contacts", 0)) > 0
        for s in steps
    )
    native_records_observed = any(
        str(s.get("native_status", "")).strip().lower() not in {"", "ok:no_contacts", "missing"}
        for s in steps
    )
    full_explicit_coverage = all(
        int(s.get("gap_unmapped_count", 0)) == 0
        and int(s.get("gap_active_projected_count", 0)) == int(s.get("gap_explicit_mapped_count", 0))
        for s in active_steps
    )
    strict_validated = all(_strict_step_validated(s) for s in active_steps)

    failures: List[str] = []
    if case.expected_contact_mode == "point" and not point_contacts_observed:
        failures.append("missing_point_contact")
    if case.expected_contact_mode == "line" and not line_contacts_observed:
        failures.append("missing_line_contact")
    if case.require_active_constraint and len(active_steps) == 0:
        failures.append("missing_active_constraint_steps")
    if len(active_steps) > 0 and not strict_validated:
        failures.append("active_steps_not_strict_validated")
    if len(active_steps) > 0 and not full_explicit_coverage:
        failures.append("active_steps_not_full_explicit_coverage")
    if len(active_steps) > 0 and not native_records_observed:
        failures.append("native_export_no_contact_records")

    pass_case = len(failures) == 0
    dominant = failures[0] if failures else "none"
    return ReferenceCaseSummary(
        case_name=case.name,
        expected_contact_mode=case.expected_contact_mode,
        require_active_constraint=bool(case.require_active_constraint),
        step_count=int(len(steps)),
        active_constraint_steps=int(len(active_steps)),
        point_contacts_observed=bool(point_contacts_observed),
        line_contacts_observed=bool(line_contacts_observed),
        native_records_observed=bool(native_records_observed),
        full_explicit_coverage_on_active_steps=bool(full_explicit_coverage),
        strict_validated_on_active_steps=bool(strict_validated),
        pass_validated_case=bool(pass_case),
        dominant_failure_reason=dominant,
        failure_reasons=failures,
    )


def _compare_case_runs(
    a: Sequence[Dict[str, Any]],
    b: Sequence[Dict[str, Any]],
) -> Tuple[bool, str, int, str, str, str]:
    if len(a) != len(b):
        return (
            False,
            f"trace_length_mismatch:{len(a)}!={len(b)}",
            0,
            "trace",
            "semantic",
            f"len_a={len(a)} len_b={len(b)}",
        )

    first_nonsemantic = ""
    for i, (ra, rb) in enumerate(zip(a, b), start=1):
        sig_a = _semantic_step_signature(ra)
        sig_b = _semantic_step_signature(rb)
        if sig_a != sig_b:
            return (
                False,
                f"semantic_step_signature_mismatch:{sig_a}!={sig_b}",
                int(i),
                "quality_gating",
                "semantic",
                "",
            )

        # Same semantic decision, but keep the first non-semantic divergence for diagnosis.
        if not first_nonsemantic:
            if str(ra.get("fp_active_rows_set", "")) != str(rb.get("fp_active_rows_set", "")):
                first_nonsemantic = (
                    f"step_{i}:constraint_domain_content:"
                    f"{ra.get('fp_active_rows_set', '')}!={rb.get('fp_active_rows_set', '')}"
                )
            elif str(ra.get("fp_native_records_set", "")) != str(rb.get("fp_native_records_set", "")):
                first_nonsemantic = (
                    f"step_{i}:native_export_content:"
                    f"{ra.get('fp_native_records_set', '')}!={rb.get('fp_native_records_set', '')}"
                )
            elif str(ra.get("fp_native_records_ordered", "")) != str(
                rb.get("fp_native_records_ordered", "")
            ):
                first_nonsemantic = (
                    f"step_{i}:native_export_order:"
                    f"{ra.get('fp_native_records_ordered', '')}!={rb.get('fp_native_records_ordered', '')}"
                )
            elif str(ra.get("native_status", "")) != str(rb.get("native_status", "")):
                first_nonsemantic = (
                    f"step_{i}:native_status_detail:"
                    f"{ra.get('native_status', '')}!={rb.get('native_status', '')}"
                )

    # Numeric traces are informative but non-semantic unless they flip quality decisions.
    lcp_a = np.asarray([float(x.get("lcp_max_abs", 0.0)) for x in a], dtype=np.float64)
    lcp_b = np.asarray([float(x.get("lcp_max_abs", 0.0)) for x in b], dtype=np.float64)
    if not np.allclose(lcp_a, lcp_b, atol=1e-6, rtol=1e-6):
        if not first_nonsemantic:
            first_nonsemantic = "numeric:lcp_trace_mismatch"
    force_a = np.asarray([float(x.get("force_norm_N", 0.0)) for x in a], dtype=np.float64)
    force_b = np.asarray([float(x.get("force_norm_N", 0.0)) for x in b], dtype=np.float64)
    if not np.allclose(force_a, force_b, atol=1e-6, rtol=1e-6):
        if not first_nonsemantic:
            first_nonsemantic = "numeric:force_trace_mismatch"

    return True, "", 0, "none", "none", first_nonsemantic


def _collect_step_fingerprints(
    *,
    force: SofaWallForceInfo,
    sim: Any,
    wall_triangle_count: int,
    contact_epsilon: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "fp_collision_constraint_raw": "",
        "fp_active_rows_ordered": "",
        "fp_active_rows_set": "",
        "fp_native_records_ordered": "",
        "fp_native_records_set": "",
        "debug_active_rows_count": 0,
        "debug_native_records_count": 0,
    }

    try:
        collision_state = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
        constraint_raw = force._read_data(collision_state, "constraint")
        out["fp_collision_constraint_raw"] = _stable_hash(str(constraint_raw or ""))
    except Exception:
        out["fp_collision_constraint_raw"] = ""

    try:
        candidates = force._collect_constraint_projection_candidates(sim)
        collision_candidates = [c for c in candidates if str(c[0]).startswith("collision.constraintProjection")]
        row_contribs: List[Dict[str, Any]] = []
        if collision_candidates:
            _src, _proj, _pos, _norm, row_contribs = max(collision_candidates, key=lambda c: float(c[3]))
        active_rows: List[Tuple[int, int]] = []
        active_rows_with_force: List[Tuple[int, int, int, int, int]] = []
        eps = float(max(contact_epsilon, 0.0))
        for rc in row_contribs:
            row_idx = rc.get("row_idx", None)
            dof_idx = rc.get("dof_idx", None)
            force_norm = float(rc.get("force_norm", 0.0))
            if not isinstance(row_idx, (int, np.integer)) or int(row_idx) < 0:
                continue
            if not isinstance(dof_idx, (int, np.integer)) or int(dof_idx) < 0:
                continue
            if not np.isfinite(force_norm) or force_norm <= eps:
                continue
            force_vec = np.asarray(rc.get("force_vec", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape((3,))
            qx, qy, qz = _quantize_xyz(force_vec)
            active_rows.append((int(row_idx), int(dof_idx)))
            active_rows_with_force.append((int(row_idx), int(dof_idx), qx, qy, qz))
        active_rows.sort()
        active_rows_with_force.sort()
        out["debug_active_rows_count"] = int(len(active_rows))
        out["fp_active_rows_ordered"] = _stable_hash(active_rows_with_force)
        out["fp_active_rows_set"] = _stable_hash(active_rows)
    except Exception:
        pass

    try:
        native_records, _native_count, _meta = force._extract_contact_records_from_native_export(
            sim,
            wall_triangle_count=int(max(wall_triangle_count, 0)),
        )
        ordered_keys: List[Tuple[Any, ...]] = []
        for rec in native_records:
            wall_point = rec.get("wall_point", None)
            wall_key = _quantize_xyz(wall_point) if wall_point is not None else (0, 0, 0)
            tri = rec.get("wall_triangle_id", None)
            row = rec.get("constraint_row_index", None)
            dof = rec.get("collision_dof_index", None)
            ordered_keys.append(
                (
                    str(rec.get("source_node_tag", "")),
                    int(rec.get("model_side", -1))
                    if isinstance(rec.get("model_side", None), (int, np.integer))
                    else -1,
                    int(rec.get("contact_local_index", -1))
                    if isinstance(rec.get("contact_local_index", None), (int, np.integer))
                    else -1,
                    int(row) if isinstance(row, (int, np.integer)) else -1,
                    int(dof) if isinstance(dof, (int, np.integer)) else -1,
                    int(tri) if isinstance(tri, (int, np.integer)) else -1,
                    wall_key,
                )
            )
        sorted_keys = sorted(ordered_keys)
        out["debug_native_records_count"] = int(len(ordered_keys))
        out["fp_native_records_ordered"] = _stable_hash(ordered_keys)
        out["fp_native_records_set"] = _stable_hash(sorted_keys)
    except Exception:
        pass

    return out


def _run_single_case(
    *,
    case: ReferenceCaseConfig,
    tool_ref: str,
    mesh_path: Path,
    units: ForceUnitsConfig,
    contact_epsilon: float,
    plugin_path: Optional[str],
) -> List[Dict[str, Any]]:
    intervention = _build_case_intervention(case=case, tool_ref=tool_ref, mesh_path=mesh_path)
    intervention.reset(seed=int(case.seed))
    sim = intervention.simulation
    if case.enable_intrusive_lcp:
        try:
            sim.root.LCP.build_lcp.value = True
            sim.root.LCP.computeConstraintForces.value = True
        except Exception:
            pass

    force = SofaWallForceInfo(
        intervention,
        mode="constraint_projected_si_validated",
        required=False,
        contact_epsilon=float(contact_epsilon),
        plugin_path=plugin_path,
        units=units,
    )

    steps: List[Dict[str, Any]] = []
    action = np.asarray([[case.action_insert, case.action_rotate]], dtype=np.float32)
    for step_idx in range(1, int(case.steps) + 1):
        intervention.step(action)
        force.step()
        info = force.info
        point_contacts, line_contacts = _collision_contact_counts(intervention)
        step_fp = _collect_step_fingerprints(
            force=force,
            sim=sim,
            wall_triangle_count=int(force._wall_segment_count),  # noqa: SLF001
            contact_epsilon=float(contact_epsilon),
        )
        steps.append(
            {
                "step": int(step_idx),
                "quality_tier": str(info.get("wall_force_quality_tier", "")),
                "status": str(info.get("wall_force_status", "")),
                "association_method": str(info.get("wall_force_association_method", "")),
                "association_coverage": float(
                    info.get("wall_force_association_coverage", float("nan"))
                ),
                "association_explicit_force_coverage": float(
                    info.get("wall_force_association_explicit_force_coverage", float("nan"))
                ),
                "association_ordering_stable": bool(
                    info.get("wall_force_association_ordering_stable", False)
                ),
                "active_constraint_step": bool(info.get("wall_force_active_constraint_step", False)),
                "lcp_max_abs": float(info.get("wall_lcp_max_abs", float("nan"))),
                "force_norm_N": float(info.get("wall_total_force_norm_N", float("nan"))),
                "gap_active_projected_count": int(
                    info.get("wall_force_gap_active_projected_count", 0)
                ),
                "gap_explicit_mapped_count": int(
                    info.get("wall_force_gap_explicit_mapped_count", 0)
                ),
                "gap_unmapped_count": int(info.get("wall_force_gap_unmapped_count", 0)),
                "gap_dominant_class": str(info.get("wall_force_gap_dominant_class", "none")),
                "gap_contact_mode": str(info.get("wall_force_gap_contact_mode", "none")),
                "native_status": str(info.get("wall_native_contact_export_status", "")),
                "native_available": bool(info.get("wall_native_contact_export_available", False)),
                "point_contacts": int(point_contacts),
                "line_contacts": int(line_contacts),
                "fp_collision_constraint_raw": str(step_fp.get("fp_collision_constraint_raw", "")),
                "fp_active_rows_ordered": str(step_fp.get("fp_active_rows_ordered", "")),
                "fp_active_rows_set": str(step_fp.get("fp_active_rows_set", "")),
                "fp_native_records_ordered": str(step_fp.get("fp_native_records_ordered", "")),
                "fp_native_records_set": str(step_fp.get("fp_native_records_set", "")),
                "debug_active_rows_count": int(step_fp.get("debug_active_rows_count", 0)),
                "debug_native_records_count": int(step_fp.get("debug_native_records_count", 0)),
            }
        )
    return steps


def _write_steps_csv(path: Path, steps: Iterable[Dict[str, Any]]) -> None:
    rows = list(steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "quality_tier",
        "status",
        "association_method",
        "association_coverage",
        "association_explicit_force_coverage",
        "association_ordering_stable",
        "active_constraint_step",
        "lcp_max_abs",
        "force_norm_N",
        "gap_active_projected_count",
        "gap_explicit_mapped_count",
        "gap_unmapped_count",
        "gap_dominant_class",
        "gap_contact_mode",
        "native_status",
        "native_available",
        "point_contacts",
        "line_contacts",
        "debug_active_rows_count",
        "debug_native_records_count",
        "fp_collision_constraint_raw",
        "fp_active_rows_ordered",
        "fp_active_rows_set",
        "fp_native_records_ordered",
        "fp_native_records_set",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_steps_json(path: Path, steps: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(steps), f, indent=2, sort_keys=True)


def run_reference_scene_suite(
    *,
    tool_ref: str = "TestModel_StandardJ035/StandardJ035_PTFE",
    output_dir: Optional[Path] = None,
    base_seed: int = 123,
    plugin_path: Optional[str] = None,
    contact_epsilon: float = 1e-7,
    units: Optional[ForceUnitsConfig] = None,
) -> ReferenceSuiteReport:
    run_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("results/force_reference_scene")
        / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    units_cfg = units or ForceUnitsConfig(length_unit="mm", mass_unit="kg", time_unit="s")
    mesh_plane = run_dir / "mesh_point_plane.obj"
    mesh_tube = run_dir / "mesh_line_tube.obj"
    _write_planar_mesh(mesh_plane)
    _write_tube_mesh(
        mesh_tube,
        radius_mm=0.6,
        length_mm=40.0,
        segments=12,
        rings=20,
    )

    cases = [
        ReferenceCaseConfig(
            name="point_vs_triangle",
            expected_contact_mode="point",
            require_active_constraint=False,
            steps=80,
            action_insert=0.3,
            action_rotate=0.0,
            seed=int(base_seed),
            enable_intrusive_lcp=False,
            line_offset_z=0.15,
        ),
        ReferenceCaseConfig(
            name="line_vs_triangle",
            expected_contact_mode="line",
            require_active_constraint=True,
            steps=80,
            action_insert=0.2,
            action_rotate=0.0,
            seed=int(base_seed),
            enable_intrusive_lcp=True,
            line_offset_z=0.55,
        ),
    ]

    case_reports: List[ReferenceCaseReport] = []
    for case in cases:
        mesh = mesh_plane if case.name == "point_vs_triangle" else mesh_tube
        steps_a = _run_single_case(
            case=case,
            tool_ref=tool_ref,
            mesh_path=mesh,
            units=units_cfg,
            contact_epsilon=contact_epsilon,
            plugin_path=plugin_path,
        )
        steps_b = _run_single_case(
            case=case,
            tool_ref=tool_ref,
            mesh_path=mesh,
            units=units_cfg,
            contact_epsilon=contact_epsilon,
            plugin_path=plugin_path,
        )
        _write_steps_csv(run_dir / f"{case.name}_run_a.csv", steps_a)
        _write_steps_csv(run_dir / f"{case.name}_run_b.csv", steps_b)
        _write_steps_json(run_dir / f"{case.name}_run_a.json", steps_a)
        _write_steps_json(run_dir / f"{case.name}_run_b.json", steps_b)

        summary_a = _summarize_case(case, steps_a)
        summary_b = _summarize_case(case, steps_b)
        (
            reproducible,
            repro_err,
            div_step,
            div_domain,
            div_kind,
            nonsemantic_div,
        ) = _compare_case_runs(steps_a, steps_b)
        pass_case = bool(summary_a.pass_validated_case and summary_b.pass_validated_case and reproducible)
        case_reports.append(
            ReferenceCaseReport(
                config=case,
                run_a=summary_a,
                run_b=summary_b,
                reproducible=bool(reproducible),
                reproducibility_error=str(repro_err),
                first_divergence_step=int(div_step),
                first_divergence_domain=str(div_domain),
                first_divergence_kind=str(div_kind),
                first_divergence_detail=str(repro_err) if repro_err else "",
                first_nonsemantic_divergence=str(nonsemantic_div),
                pass_validated_case=bool(pass_case),
            )
        )

    pass_suite = all(rep.pass_validated_case for rep in case_reports)
    external_limit_reason = ""
    external_limit_detected = False
    if not pass_suite:
        line_rep = next((r for r in case_reports if r.config.name == "line_vs_triangle"), None)
        if line_rep is not None:
            if (
                line_rep.run_a.active_constraint_steps > 0
                and line_rep.run_b.active_constraint_steps > 0
                and not line_rep.run_a.native_records_observed
                and not line_rep.run_b.native_records_observed
            ):
                external_limit_detected = True
                external_limit_reason = "native_contact_export_no_records_in_active_line_case"
            elif (
                line_rep.run_a.active_constraint_steps > 0
                and line_rep.run_b.active_constraint_steps > 0
                and (
                    not line_rep.run_a.full_explicit_coverage_on_active_steps
                    or not line_rep.run_b.full_explicit_coverage_on_active_steps
                )
            ):
                external_limit_reason = "active_line_case_without_full_explicit_coverage"
            elif line_rep.run_a.active_constraint_steps == 0 or line_rep.run_b.active_constraint_steps == 0:
                external_limit_reason = "line_case_without_active_constraints"
            else:
                external_limit_reason = "reference_suite_not_validated"
        else:
            external_limit_reason = "missing_line_case_report"

    suite_report = ReferenceSuiteReport(
        created_at=datetime.utcnow().isoformat() + "Z",
        tool_ref=tool_ref,
        output_dir=str(run_dir),
        pass_validated_suite=bool(pass_suite),
        external_limit_detected=bool(external_limit_detected),
        external_limit_reason=str(external_limit_reason),
        case_reports=case_reports,
    )

    with (run_dir / "reference_scene_report.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(suite_report), f, indent=2, sort_keys=True)
    return suite_report
