from __future__ import annotations

from pathlib import Path
from typing import Any


def _mesh_text_cube(radius_mm: float) -> str:
    r = float(radius_mm)
    return "\n".join(
        [
            f"v 0.000000 0.000000 {r:.6f}",
            f"v 0.000000 0.000000 {-r:.6f}",
            f"v {-r:.6f} 0.000000 0.000000",
            f"v {r:.6f} 0.000000 0.000000",
            f"v 0.000000 {-r:.6f} 0.000000",
            f"v 0.000000 {r:.6f} 0.000000",
            "f 1 4 6",
            "f 1 6 3",
            "f 1 3 5",
            "f 1 5 4",
            "f 2 6 4",
            "f 2 3 6",
            "f 2 5 3",
            "f 2 4 5",
            "",
        ]
    )


def _ensure_mesh(path: Path, *, radius_mm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _mesh_text_cube(radius_mm)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _mesh_text_plane(half_extent_mm: float = 100.0) -> str:
    """Create a simple plane mesh centered at origin, lying in XY plane at Z=0."""
    h = float(half_extent_mm)
    return "\n".join(
        [
            f"v {-h:.6f} {-h:.6f} 0.000000",
            f"v {h:.6f} {-h:.6f} 0.000000",
            f"v {h:.6f} {h:.6f} 0.000000",
            f"v {-h:.6f} {h:.6f} 0.000000",
            "f 1 2 3",
            "f 1 3 4",
            "",
        ]
    )


def _ensure_plane_mesh(path: Path, *, half_extent_mm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_mesh_text_plane(half_extent_mm), encoding="utf-8")


def createScene(root: Any, *, friction_mu: float = 0.0, dt_s: float = 0.01) -> Any:
    """Build a minimal sphere-on-plane validation scene.

    The scene is intentionally compact: it exercises SOFA's contact stack and
    keeps the geometry simple so force recovery can be checked independently of
    the intervention pipeline.

    Note: visualization components (OglModel, etc.) are omitted since SOFA v22.06+
    moved them to Sofa.GL.Component.Rendering3D. They're not needed for physics validation.
    """

    root.gravity = [0.0, 0.0, -9810.0]
    root.dt = float(dt_s)
    try:
        root.addObject(
            "RequiredPlugin",
            pluginName=[
                "Sofa.Component.AnimationLoop",
                "Sofa.Component.Collision.Detection.Algorithm",
                "Sofa.Component.Collision.Detection.Intersection",
                "Sofa.Component.Collision.Geometry",
                "Sofa.Component.Collision.Response.Contact",
                "Sofa.Component.Constraint.Lagrangian.Correction",
                "Sofa.Component.Constraint.Lagrangian.Solver",
                "Sofa.Component.IO.Mesh",
                "Sofa.Component.LinearSolver.Direct",
                "Sofa.Component.LinearSolver.Iterative",
                "Sofa.Component.Mapping.Linear",
                "Sofa.Component.Mapping.NonLinear",
                "Sofa.Component.Mass",
                "Sofa.Component.ODESolver.Backward",
                "Sofa.Component.SolidMechanics.Spring",
                "Sofa.Component.StateContainer",
                "Sofa.Component.Topology.Container.Dynamic",
            ],
        )
    except Exception:
        pass

    root.addObject("FreeMotionAnimationLoop")
    root.addObject("DefaultPipeline", draw="0", depth="6", verbose="1")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject(
        "LocalMinDistance",
        contactDistance=2.0,
        alarmDistance=10.0,
        angleCone=0.02,
        name="localmindistance",
    )
    root.addObject(
        "DefaultContactManager",
        name="contact_manager",
        response="FrictionContactConstraint",
        responseParams=f"mu={float(friction_mu)}",
    )
    root.addObject(
        "LCPConstraintSolver",
        name="LCP",
        mu=float(friction_mu),
        tolerance=1e-6,
        maxIt=1000,
        build_lcp=True,
        computeConstraintForces=True,
    )

    plane_mesh_path = Path(__file__).resolve().parent / "validation_sphere_on_plane_floor.obj"
    _ensure_plane_mesh(plane_mesh_path, half_extent_mm=150.0)

    # Static floor using the same mesh/topology pattern used in other SOFA scenes.
    floor = root.addChild("Floor")
    floor.addObject(
        "MeshObjLoader",
        filename=str(plane_mesh_path),
        name="loader",
    )
    floor.addObject(
        "MeshTopology",
        position="@loader.position",
        triangles="@loader.triangles",
    )
    floor.addObject("MechanicalObject", name="dofs", src="@loader")
    floor.addObject("TriangleCollisionModel", moving=False, simulated=False)
    floor.addObject("LineCollisionModel", moving=False, simulated=False)
    floor.addObject("PointCollisionModel", moving=False, simulated=False)

    sphere_mesh_path = Path(__file__).resolve().parent / "validation_sphere_on_plane.obj"
    _ensure_mesh(sphere_mesh_path, radius_mm=10.0)

    # Dynamic rigid sphere with mesh collision, following the SOFA example
    # pattern that exposes triangle/line/point proximity models to the contact
    # manager while keeping the mass on the rigid body.
    sphere = root.addChild("Sphere")
    sphere.addObject("EulerImplicitSolver")
    sphere.addObject("CGLinearSolver", iterations=100, tolerance=1e-10, threshold=1e-10)
    sphere.addObject(
        "MechanicalObject",
        name="dofs",
        template="Rigid3d",
        position=[0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 1.0],
    )
    sphere.addObject("UniformMass", totalMass=1.0)
    sphere.addObject("UncoupledConstraintCorrection")

    collision = sphere.addChild("Collision")
    collision.addObject(
        "MeshObjLoader",
        filename=str(sphere_mesh_path),
        triangulate=True,
        name="loader",
    )
    collision.addObject("MeshTopology", src="@loader")
    collision.addObject("MechanicalObject", name="dofs", src="@loader")
    collision.addObject("TriangleCollisionModel")
    collision.addObject("LineCollisionModel")
    collision.addObject("PointCollisionModel")
    collision.addObject("RigidMapping")

    return root
