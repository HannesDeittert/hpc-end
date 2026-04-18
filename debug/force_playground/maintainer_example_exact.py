# Required import for python
import argparse
import os
from pathlib import Path
import sys
import Sofa
import numpy as np
import math


USE_GUI = True


def _build_gui_ld_library_path() -> str:
    parts = []
    sofa_root = os.environ.get("SOFA_ROOT", "")
    if sofa_root:
        parts.extend(
            [
                os.path.join(sofa_root, "lib"),
                os.path.join(sofa_root, "plugins", "SofaPython3", "lib"),
            ]
        )
    parts.append("/usr/lib/x86_64-linux-gnu")
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts.extend([p for p in old.split(":") if p and p not in parts])
    return ":".join(parts)


def _reexec_for_qt_runtime() -> None:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _build_gui_ld_library_path()
    env["STEVE_SOFA_GUI_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def _resolve_mesh_path(mesh_filename):
    mesh_name = Path(mesh_filename).name
    candidates = [
        Path(mesh_filename),
        Path(__file__).resolve().parent / "mesh" / mesh_name,
    ]

    sofa_root = os.environ.get("SOFA_ROOT")
    if sofa_root:
        candidates.append(Path(sofa_root) / "share" / "sofa" / "mesh" / mesh_name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return mesh_filename


# Main function taking a boolean as parameter to choose whether or not to use the GUI
def main(use_gui=True, steps=10, gui_backend="imgui"):
    root = Sofa.Core.Node("root")
    createScene(root)
    if hasattr(Sofa.Simulation, "initRoot"):
        Sofa.Simulation.initRoot(root)
    else:
        Sofa.Simulation.init(root)

    if not use_gui:
        for iteration in range(int(steps)):
            Sofa.Simulation.animate(root, root.dt.value)
        Sofa.Simulation.unload(root)
    else:
        backend = str(gui_backend)
        if backend == "imgui":
            try:
                import SofaImGui  # noqa: F401
            except ModuleNotFoundError:
                print(
                    "[maintainer_example_exact] SofaImGui not found. Falling back to qglviewer."
                )
                backend = "qglviewer"
        try:
            from Sofa import Gui as SofaGui
        except ImportError as exc:
            message = str(exc)
            if (
                "libQGLViewer.so" in message
                and os.environ.get("STEVE_SOFA_GUI_REEXEC") != "1"
            ):
                print(
                    "[maintainer_example_exact] Qt runtime mismatch detected. "
                    "Retrying once with SOFA Qt library order."
                )
                _reexec_for_qt_runtime()
            raise
        SofaGui.GUIManager.Init("myscene", backend)
        SofaGui.GUIManager.createGUI(root, __file__)
        SofaGui.GUIManager.SetDimension(1080, 1080)
        SofaGui.GUIManager.MainLoop(root)
        SofaGui.GUIManager.closeGUI()
        Sofa.Simulation.unload(root)


def createScene(root):
    root.gravity = [0, -9.81, 0]
    root.dt = 0.02

    root.addObject(
        "RequiredPlugin",
        pluginName=[
            "Sofa.Component.Collision.Detection.Algorithm",
            "Sofa.Component.Collision.Detection.Intersection",
            "Sofa.Component.Collision.Geometry",
            "Sofa.Component.Collision.Response.Contact",
            "Sofa.Component.Constraint.Projective",
            "Sofa.Component.IO.Mesh",
            "Sofa.Component.LinearSolver.Iterative",
            "Sofa.Component.Mapping.Linear",
            "Sofa.Component.Mass",
            "Sofa.Component.ODESolver.Backward",
            "Sofa.Component.SolidMechanics.FEM.Elastic",
            "Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Dynamic",
            "Sofa.Component.Visual",
            "Sofa.GL.Component.Rendering3D",
            "Sofa.Component.Constraint.Lagrangian.Correction",
            "Sofa.Component.Constraint.Lagrangian.Solver",
            "Sofa.Component.MechanicalLoad",
            "Sofa.Component.LinearSolver.Direct",
            "Sofa.Component.AnimationLoop",
        ],
    )

    root.addObject("FreeMotionAnimationLoop")
    # Constraint solver computing the constraint/contact forces, stored in the constraint space (normal , tangential_1, tangential_2)
    constraint_solver = root.addObject(
        "GenericConstraintSolver",
        maxIterations=1000,
        tolerance=1e-6,
        computeConstraintForces=True,
    )

    root.addObject(
        "VisualStyle",
        displayFlags="showCollisionModels hideVisualModels showForceFields",
    )
    root.addObject("CollisionPipeline", name="collision_pipeline")
    root.addObject("BruteForceBroadPhase", name="broad_phase")
    root.addObject("BVHNarrowPhase", name="narrow_phase")
    root.addObject("DiscreteIntersection")
    root.addObject(
        "CollisionResponse",
        name="collision_response",
        response="FrictionContactConstraint",
        responseParams="mu=0.1",
    )

    root.addObject(
        "MeshOBJLoader",
        name="load_liver_surface",
        filename=_resolve_mesh_path("mesh/liver-smooth.obj"),
    )

    liver = root.addChild("Liver")
    liver.addObject(
        "EulerImplicitSolver", name="cg_odesolver", rayleighStiffness=0.1, rayleighMass=0.1
    )
    liver.addObject(
        "SparseLDLSolver", name="linear_solver", template="CompressedRowSparseMatrixMat3x3d"
    )
    liver.addObject(
        "MeshGmshLoader",
        name="loader_liver_volume",
        filename=_resolve_mesh_path("mesh/liver.msh"),
    )
    liver.addObject("TetrahedronSetTopologyContainer", name="topo", src="@loader_liver_volume")
    # Liver MechanicalObject where the constraint/contact forces will be stored in the (x,y,z) coordinate system
    liverMO = liver.addObject("MechanicalObject", name="dofs", src="@loader_liver_volume")
    liver.addObject("TetrahedronSetGeometryAlgorithms", template="Vec3d", name="geom_algo")
    liver.addObject("DiagonalMass", name="mass", massDensity=1.0)
    liver.addObject(
        "TetrahedralCorotationalFEMForceField",
        template="Vec3d",
        name="FEM",
        method="large",
        poissonRatio=0.3,
        youngModulus=3000,
        computeGlobalMatrix=False,
    )
    try:
        liver.addObject("FixedProjectiveConstraint", name="fixed_constraint", indices=[3, 39, 64])
    except Exception:
        print(
            "[maintainer_example_exact] FixedProjectiveConstraint unavailable. "
            "Using FixedConstraint."
        )
        liver.addObject("FixedConstraint", name="fixed_constraint", indices=[3, 39, 64])
    liver.addObject("LinearSolverConstraintCorrection")

    # Forcefield only used for visualization purposes (of the contact fborces)
    contactVisu = liver.addChild("RenderingContactForces")
    contactVisu.addObject("VisualStyle", displayFlags="showVisualModels")
    renderingForces = None
    try:
        renderingForces = contactVisu.addObject(
            "VisualVectorField",
            name="drawing_contact_forces",
            vectorScale="100",
            vector="@../dofs.lambda",
            position="@../dofs.position",
            color="orange",
            drawMode="Arrow",
        )
    except Exception:
        print(
            "[maintainer_example_exact] VisualVectorField unavailable. "
            "Running without force arrows."
        )

    visu = liver.addChild("Visu")
    visu.addObject("OglModel", name="visual_model", src="@../../load_liver_surface")
    visu.addObject("BarycentricMapping", name="visual_mapping", input="@../dofs", output="@visual_model")

    surf = liver.addChild("Surf")
    surf.addObject(
        "SphereLoader",
        name="loader_sphere_model",
        filename=_resolve_mesh_path("mesh/liver.sph"),
    )
    surf.addObject("MechanicalObject", name="spheres", position="@loader_sphere_model.position")
    surf.addObject("SphereCollisionModel", name="collision_model", listRadius="@loader_sphere_model.listRadius")
    surf.addObject("BarycentricMapping", name="collision_mapping", input="@../dofs", output="@spheres")

    particle = root.addChild("Particle")
    particle.addObject("EulerImplicitSolver")
    particle.addObject("CGLinearSolver", threshold=1e-09, tolerance=1e-09, iterations=200)
    # Particle MechanicalObject where the constraint/contact forces will be stored in the (x,y,z) coordinate system
    particleMO = particle.addObject(
        "MechanicalObject",
        showObject=True,
        position=[-2, 10, 0, 0, 0, 0, 1],
        name=f"particle_DoFs",
        template="Rigid3d",
    )
    particle.addObject("UniformMass", totalMass=1)
    particle.addObject("ConstantForceField", name="CFF", totalForce=[0, -1, 0, 0, 0, 0])
    particle.addObject("SphereCollisionModel", name="SCM", radius=1.0)
    particle.addObject("UncoupledConstraintCorrection")

    # Python controller accessing and displaying the contact forces in the ConstantForceField
    root.addObject(
        AccessContactForces(
            "AccessContactForces",
            name="AccessContactForces",
            constraint_solver=constraint_solver,
            soft_liver=liverMO,
            forces_visu=renderingForces,
            root_node=root,
        )
    )


class AccessContactForces(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.constraint_solver = kwargs.get("constraint_solver")
        self.soft_liver = kwargs.get("soft_liver")
        self.forces_visu = kwargs.get("forces_visu")
        self.root_node = kwargs.get("root_node")
        # Initialize the rendered vector with zero vec<Vec3>
        if self.forces_visu is not None:
            self.forces_visu.vector.value = np.zeros((181, 3))

    def onAnimateEndEvent(self, event):
        lambda_vector = self.constraint_solver.constraintForces.value
        # If there is a contact
        if len(lambda_vector) > 0:
            print(
                f"At time = {round(self.root_node.time.value,3)}, forces in the contact space (n, t1, t2) equals:\n  {lambda_vector} "
            )

            # Compute the inverse (reaction force)
            xyz_reaction = -np.asarray(
                self.soft_liver.getData("lambda").value, dtype=np.float32
            )
            print(
                f"At time = {round(self.root_node.time.value,3)}, forces in the (x,y,z) coordinate system equals:\n  {xyz_reaction} "
            )
            if self.forces_visu is not None:
                self.forces_visu.vector.value = xyz_reaction

                # Scale automatically the displayed force vector
                fact = float(np.max(np.abs(np.asarray(lambda_vector, dtype=np.float32))))
                if fact > 0.0:
                    self.forces_visu.vectorScale.value = 10 / fact

        # If no contact
        else:
            if self.forces_visu is not None:
                self.forces_visu.vector.value = np.zeros((181, 3))


# Function used only if this script is called from a python environment
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maintainer contact-force example with optional headless mode."
    )
    parser.add_argument("--gui", dest="gui", action="store_true")
    parser.add_argument("--no-gui", dest="gui", action="store_false")
    parser.set_defaults(gui=USE_GUI)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backend", choices=["imgui", "qglviewer"], default="imgui")
    args = parser.parse_args()
    main(use_gui=args.gui, steps=args.steps, gui_backend=args.backend)
