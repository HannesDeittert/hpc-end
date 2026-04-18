from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from steve_recommender.adapters import eve
from steve_recommender.bench.straight_eps import StraightEps
from steve_recommender.devices import make_device
from steve_recommender.evaluation.info_collectors import SofaWallForceInfo

from .config import ForcePlaygroundConfig


@dataclass
class ForcePlaygroundScene:
    cfg: ForcePlaygroundConfig
    intervention: Any
    simulation: Any
    force_info: SofaWallForceInfo
    wall_vertices: np.ndarray
    wall_triangles: np.ndarray
    wall_centroids: np.ndarray
    wall_normals: np.ndarray
    wall_reference_normal: np.ndarray
    camera_preset: str
    camera_position: np.ndarray
    camera_look_at: np.ndarray
    mesh_path: Path
    dt_s: float


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape((3,))
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (arr / n).astype(np.float32)


def _write_plane_mesh(path: Path, *, width_mm: float, height_mm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    half_w = float(width_mm) * 0.5
    half_h = float(height_mm) * 0.5
    text = "\n".join(
        [
            f"v {-half_w:.6f} {-half_h:.6f} 0.0",
            f"v {half_w:.6f} {-half_h:.6f} 0.0",
            f"v {half_w:.6f} {half_h:.6f} 0.0",
            f"v {-half_w:.6f} {half_h:.6f} 0.0",
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
    verts = []
    for ring in range(rings + 1):
        x = float(ring) * float(length_mm) / float(rings)
        for seg in range(segments):
            angle = (2.0 * math.pi * float(seg)) / float(segments)
            y = float(radius_mm) * math.cos(angle)
            z = float(radius_mm) * math.sin(angle)
            verts.append((x, y, z))

    faces = []
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


def _make_device(cfg: ForcePlaygroundConfig) -> Any:
    if cfg.probe == "guidewire":
        return make_device(cfg.tool_ref)

    # v1 rigid probe: intentionally high stiffness straight rod.
    return StraightEps(
        name="rigid_probe",
        velocity_limit=(35.0, 3.14),
        length=300.0,
        tip_length=10.0,
        tip_outer_diameter=0.7,
        straight_outer_diameter=0.89,
        young_modulus_tip=5.0e6,
        young_modulus_straight=5.0e6,
        mass_density_tip=2.1e-5,
        mass_density_straight=2.1e-5,
        beams_per_mm_tip=2.0,
        beams_per_mm_straight=0.3,
        collis_edges_per_mm_tip=2.0,
        collis_edges_per_mm_straight=0.2,
        spire_diameter=0.3,
    )


def _build_scene_intervention(cfg: ForcePlaygroundConfig, mesh_path: Path) -> Any:
    device = _make_device(cfg)
    if cfg.scene == "plane_wall":
        probe_diameter = float(
            getattr(
                device,
                "straight_outer_diameter",
                getattr(device, "tip_outer_diameter", 0.89),
            )
        )
        probe_radius = 0.5 * probe_diameter
        # Keep a small positive gap to avoid pre-penetration at step 0 while
        # still allowing force-driven wall contact during oracle runs.
        z_offset_mm = probe_radius + 0.02
        centerline = np.stack(
            [
                np.linspace(-80.0, 120.0, 240, dtype=np.float32),
                np.zeros((240,), dtype=np.float32),
                np.full((240,), z_offset_mm, dtype=np.float32),
            ],
            axis=1,
        )
        insertion = (-80.0, 0.0, z_offset_mm)
        approx_radius = 1.0
    elif cfg.scene == "tube_wall":
        centerline = np.stack(
            [
                np.linspace(0.0, cfg.mesh.tube_length_mm, 220, dtype=np.float32),
                np.full((220,), 0.08, dtype=np.float32),
                np.full((220,), 0.55, dtype=np.float32),
            ],
            axis=1,
        )
        insertion = (0.0, 0.08, 0.55)
        approx_radius = float(cfg.mesh.tube_radius_mm)
    else:
        raise ValueError(f"Unsupported scene: {cfg.scene}")

    branch = eve.intervention.vesseltree.Branch(name="main", coordinates=centerline)
    vessel_tree = eve.intervention.vesseltree.FromMesh(
        mesh=str(mesh_path),
        insertion_position=insertion,
        insertion_direction=(1.0, 0.0, 0.0),
        branch_list=[branch],
        approx_branch_radii=approx_radius,
        rotation_yzx_deg=(0.0, 0.0, 0.0),
    )

    simulation = eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter(
        friction=float(cfg.friction)
    )
    # Important for --show-sofa:
    # The simulation graph is built during intervention.reset().
    # If visual flags are enabled only later (inside SofaPygame), camera nodes are
    # missing because stEVE does not rebuild on init_visual_nodes flips alone.
    # So we enable visual-node construction upfront when scene-view is requested.
    if bool(cfg.show_sofa):
        simulation.init_visual_nodes = True
        simulation.display_size = (600, 860)

    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=float(cfg.image_frequency_hz),
        image_rot_zx=[0.0, 0.0],
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
    if bool(cfg.show_sofa):
        simulation.target_size = float(target.threshold)
        simulation.interim_target_size = float(target.threshold)
    intervention.make_non_mp()
    return intervention


def _configure_contact_detection(simulation: Any, *, alarm_distance: float, contact_distance: float) -> None:
    lmd = getattr(getattr(simulation, "root", None), "localmindistance", None)
    if lmd is None:
        return
    _set_data_field(lmd, "alarmDistance", float(alarm_distance))
    _set_data_field(lmd, "contactDistance", float(contact_distance))


def _compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if vertices.size == 0 or triangles.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    normals = np.zeros((triangles.shape[0], 3), dtype=np.float32)
    for i, tri in enumerate(triangles):
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(b - a, c - a)
        normals[i] = _normalize(n)
    return normals


def _reference_wall_normal(cfg: ForcePlaygroundConfig, wall_normals: np.ndarray) -> np.ndarray:
    if wall_normals.size == 0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if cfg.scene == "plane_wall":
        mean_n = np.mean(wall_normals, axis=0)
        ref = _normalize(mean_n)
        # Keep deterministic orientation: positive z for plane scene.
        if ref[2] < 0.0:
            ref = -ref
        return ref.astype(np.float32)
    # tube: no unique global normal; use mean as reference for aggregated oracle-disabled views.
    return _normalize(np.mean(wall_normals, axis=0)).astype(np.float32)


def _orthonormal_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _normalize(normal)
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(seed, n))) > 0.95:
        seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    t1 = seed - float(np.dot(seed, n)) * n
    t1 = _normalize(t1)
    t2 = _normalize(np.cross(n, t1))
    return t1, t2


def select_camera_pose(
    cfg: ForcePlaygroundConfig,
    wall_vertices: np.ndarray,
    wall_reference_normal: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return deterministic camera preset + pose for visual stability."""
    if cfg.camera_preset == "auto":
        preset = "plane_oblique" if cfg.scene == "plane_wall" else "tube_oblique"
    else:
        preset = str(cfg.camera_preset)

    if wall_vertices.size > 0:
        center = np.mean(wall_vertices, axis=0).astype(np.float32)
        bounds = np.max(wall_vertices, axis=0) - np.min(wall_vertices, axis=0)
        diag = float(np.linalg.norm(bounds))
    else:
        center = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        diag = 0.0
    dist = max(80.0, 1.5 * diag)

    n = _normalize(wall_reference_normal)
    t1, t2 = _orthonormal_tangent_basis(n)

    if preset == "plane_front":
        position = center + n * dist
    elif preset == "plane_oblique":
        position = center + n * (0.95 * dist) + t1 * (0.60 * dist) - t2 * (0.75 * dist)
    elif preset == "tube_oblique":
        position = center + n * (0.65 * dist) + t1 * (0.85 * dist) - t2 * (0.80 * dist)
    else:
        raise ValueError(f"Unsupported camera preset: {preset}")

    return preset, position.astype(np.float32), center.astype(np.float32)


def _set_data_field(obj: Any, name: str, value: Any) -> bool:
    try:
        data = getattr(obj, name)
        if hasattr(data, "value"):
            data.value = value
        else:
            setattr(obj, name, value)
        return True
    except Exception:
        pass
    try:
        data = obj.findData(name)
        data.value = value
        return True
    except Exception:
        return False


def apply_camera_pose(
    simulation: Any,
    *,
    position: np.ndarray,
    look_at: np.ndarray,
) -> bool:
    camera = getattr(simulation, "camera", None)
    if camera is None:
        return False

    pos = np.asarray(position, dtype=np.float32).reshape((3,))
    ctr = np.asarray(look_at, dtype=np.float32).reshape((3,))

    # SOFA Camera python bindings are sensitive to assignment types.
    # Use plain python lists/scalars to avoid binding crashes.
    ok = True
    ok &= _set_data_field(camera, "position", [float(x) for x in pos])
    ok &= _set_data_field(camera, "lookAt", [float(x) for x in ctr])

    # Keep clipping stable for very flat geometries.
    dist = float(np.linalg.norm(pos - ctr))
    z_near = max(0.5, 0.02 * dist)
    z_far = max(50.0, 8.0 * dist)
    _set_data_field(camera, "zNear", z_near)
    _set_data_field(camera, "zFar", z_far)
    return bool(ok)


def build_scene(cfg: ForcePlaygroundConfig, run_dir: Path) -> ForcePlaygroundScene:
    mesh_dir = run_dir / "meshes"
    if cfg.scene == "plane_wall":
        mesh_path = mesh_dir / "plane_wall.obj"
        _write_plane_mesh(
            mesh_path,
            width_mm=float(cfg.mesh.plane_width_mm),
            height_mm=float(cfg.mesh.plane_height_mm),
        )
    elif cfg.scene == "tube_wall":
        mesh_path = mesh_dir / "tube_wall.obj"
        _write_tube_mesh(
            mesh_path,
            radius_mm=float(cfg.mesh.tube_radius_mm),
            length_mm=float(cfg.mesh.tube_length_mm),
            segments=int(cfg.mesh.tube_segments),
            rings=int(cfg.mesh.tube_rings),
        )
    else:
        raise ValueError(f"Unsupported scene: {cfg.scene}")

    intervention = _build_scene_intervention(cfg, mesh_path)
    intervention.reset(seed=int(cfg.seed))
    simulation = intervention.simulation
    _configure_contact_detection(
        simulation,
        alarm_distance=float(cfg.alarm_distance),
        contact_distance=float(cfg.contact_distance),
    )

    dt_s = 1.0 / float(intervention.fluoroscopy.image_frequency)

    force_info = SofaWallForceInfo(
        intervention,
        mode="constraint_projected_si_validated",
        required=False,
        contact_epsilon=float(cfg.contact_epsilon),
        plugin_path=cfg.plugin_path,
        units=cfg.units,
        constraint_dt_s=dt_s,
    )
    force_info.step()

    wall_vertices, wall_triangles, wall_centroids = force_info._read_vessel_wall_geometry(simulation)  # noqa: SLF001
    wall_normals = _compute_normals(wall_vertices, wall_triangles)
    wall_reference_normal = _reference_wall_normal(cfg, wall_normals)
    camera_preset, camera_position, camera_look_at = select_camera_pose(
        cfg,
        wall_vertices,
        wall_reference_normal,
    )
    if bool(cfg.show_sofa):
        apply_camera_pose(
            simulation,
            position=camera_position,
            look_at=camera_look_at,
        )

    return ForcePlaygroundScene(
        cfg=cfg,
        intervention=intervention,
        simulation=simulation,
        force_info=force_info,
        wall_vertices=wall_vertices,
        wall_triangles=wall_triangles,
        wall_centroids=wall_centroids,
        wall_normals=wall_normals,
        wall_reference_normal=wall_reference_normal,
        camera_preset=camera_preset,
        camera_position=camera_position,
        camera_look_at=camera_look_at,
        mesh_path=mesh_path,
        dt_s=dt_s,
    )
