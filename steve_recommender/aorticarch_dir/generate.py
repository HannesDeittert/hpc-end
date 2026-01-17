import os
import json
import shutil
from pathlib import Path
from typing import Tuple

from steve_recommender.adapters import eve

ArchType = eve.intervention.vesseltree.ArchType
AorticArch = eve.intervention.vesseltree.AorticArch

# Configure how many unique trees and where to put them
target_root = Path(__file__).parent / "vesseltrees"
num_trees = 20

def make_tree_config(index: int) -> dict:
    """
    Define a deterministic set of parameters for each tree.
    Here we cycle through ArchType values and vary the seed.
    """
    arch_types = list(ArchType)
    arch_type = arch_types[index % len(arch_types)]
    seed = 1000 + index
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scaling: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    omit_axis = None
    return {
        "name": f"Tree_{index:02d}",
        "arch_type": arch_type,
        "seed": seed,
        "rotation_yzx_deg": rotation,
        "scaling_xyzd": scaling,
        "omit_axis": omit_axis,
    }

# Template for the per-tree Python file
template_py = '''
from steve_recommender.adapters import eve

VesselTree = eve.intervention.vesseltree.VesselTree
AorticArch = eve.intervention.vesseltree.aorticarch.AorticArch
ArchType = eve.intervention.vesseltree.aorticarch.ArchType

class {class_name}(VesselTree):
    """
    Deterministic VesselTree alias of AorticArch with fixed parameters.
    """
    def __init__(self) -> None:
        super().__init__(
            arch_type=ArchType.{arch_type},
            seed={seed},
            rotation_yzx_deg={rotation},
            scaling_xyzd={scaling},
            omit_axis={omit_axis!r},
        )
        # Build branches and coordinate space once
        self.reset()
'''

# Generate static vessel trees
for i in range(num_trees):
    cfg = make_tree_config(i)
    tree_name = cfg["name"]
    tree_dir = target_root / tree_name
    mesh_dir = tree_dir / "mesh"
    py_file = tree_dir / f"{tree_name}.py"
    desc_file = tree_dir / "description.json"

    # Create directories
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate AorticArch and build it
    arch = AorticArch(
        arch_type=cfg["arch_type"],
        seed=cfg["seed"],
        rotation_yzx_deg=cfg["rotation_yzx_deg"],
        scaling_xyzd=cfg["scaling_xyzd"],
        omit_axis=cfg["omit_axis"],
    )
    arch.reset()

    # Copy the simulation mesh and the visualization mesh, preserving extensions
    sim_src = Path(arch.mesh_path)
    vis_src = Path(arch.visu_mesh_path or arch.mesh_path)
    sim_ext = sim_src.suffix.lower()
    vis_ext = vis_src.suffix.lower()
    shutil.copy(sim_src, mesh_dir / f"simulationmesh{sim_ext}")
    shutil.copy(vis_src, mesh_dir / f"visumesh{vis_ext}")

    # Write description.json with mesh paths
    desc = {
        "arch_type": cfg["arch_type"].value,
        "seed": cfg["seed"],
        "rotation_yzx_deg": cfg["rotation_yzx_deg"],
        "scaling_xyzd": cfg["scaling_xyzd"],
        "omit_axis": cfg["omit_axis"],
        "simulation_mesh": f"mesh/simulationmesh{sim_ext}",
        "visu_mesh": f"mesh/visumesh{vis_ext}",
    }
    desc_file.write_text(json.dumps(desc, indent=4))

    # Write the Python alias class file
    template_py_file = template_py.format(
        class_name=tree_name,
        arch_type=cfg["arch_type"].name,
        seed=cfg["seed"],
        rotation=cfg["rotation_yzx_deg"],
        scaling=cfg["scaling_xyzd"],
        omit_axis=cfg["omit_axis"],
    )
    py_file.write_text(template_py_file)

print(f"Generated {num_trees} static VesselTree folders under {target_root}")
