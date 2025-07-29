import os
import shutil

from eve.intervention.vesseltree import ArchType, AorticArch

# -------------------------------------------------------------------
# Configuration: adjust as needed
# -------------------------------------------------------------------
ARCH_TYPE = ArchType.I
SEED = 123456
# Output mesh directory (will be created if missing)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "meshes")
MESH_FILENAME = "aorta_fixed_I.vtp"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build a reproducible AorticArch instance
def create_vessel_tree(
    arch_type: ArchType = ARCH_TYPE,
    seed: int = SEED,
) -> AorticArch:
    """
    Creates an AorticArch with fixed parameters and saves its mesh.

    Returns:
        An AorticArch instance with its mesh_path set to a persistent file.
    """
    # Instantiate the arch
    arch = AorticArch(arch_type=arch_type, seed=seed)

    # Trigger lazy mesh generation
    temp_mesh = arch.mesh_path

    # Copy to persistent location
    final_path = os.path.join(OUTPUT_DIR, MESH_FILENAME)
    if not os.path.exists(final_path):
        shutil.copy(temp_mesh, final_path)

    # Override internal mesh_path so future calls use the saved file
    arch._mesh_path = final_path
    return arch

# Create a module‐level vessel_tree for easy import
vessel_tree = create_vessel_tree()
mesh_path = vessel_tree.mesh_path  # path to the saved mesh
