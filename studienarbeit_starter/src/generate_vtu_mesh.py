"""Mesh generation from fiber distributions for the 4C solver."""

import csv
from pathlib import Path

import gmsh
import meshio
import numpy as np

# ================================================================
# Constants
# ================================================================
SIZE_TOLERANCE_FACTOR = 1.05  # Tolerance factor for identifying clipped fibers
BOUNDARY_TOLERANCE = 1e-3  # Tolerance for detecting nodes on boundaries


# ================================================================
# Helper Functions for Boundary Detection
# ================================================================
def at_min(points, axis):
    """Check if points are at minimum boundary (e.g. x = 0)."""
    return np.abs(points[:, axis]) < BOUNDARY_TOLERANCE


def at_max(points, axis, size):
    """Check if points are at maximum boundary (e.g. x = x_max)."""
    return np.abs(points[:, axis] - size) < BOUNDARY_TOLERANCE


# ================================================================
# CSV Reading
# ================================================================
def load_fiber_data(csv_path):
    """Load fiber centers and radii from CSV file."""
    fibers = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x_coordinate"])
            y = float(row["y_coordinate"])
            radius = float(row["radius"])
            fibers.append((x, y, radius))

    return fibers


# ================================================================
# GMSH Geometry Creation
# ================================================================
def create_2D_geometry(fibers, rve_size_x, rve_size_y):
    """Create 2D geometry: matrix with fiber cutouts.

    Args:
        fibers: List of (x, y, radius) tuples.
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
    """
    # Clipping box to cut fibers at RVE boundaries
    clip_box = gmsh.model.occ.addRectangle(0, 0, 0, rve_size_x, rve_size_y)

    # Create and clip fibers
    clipped_fiber_tags = []
    for x, y, radius in fibers:
        fiber = gmsh.model.occ.addDisk(x, y, 0, radius, radius)

        # Cut away parts of the fibers that are outside the RVE boundaries
        clipped_fibers = gmsh.model.occ.intersect(
            [(2, fiber)],           # Object to clip
            [(2, clip_box)],        # Clipping part
            removeObject = True,    # Delete original fiber (keep only clipped fiber)
            removeTool = False)     # Keep clip_box for next fiber

        # Collect clipped fiber tags
        for dim, tag in clipped_fibers[0]:
            clipped_fiber_tags.append(tag)

    # Create square matrix
    matrix = gmsh.model.occ.addRectangle(0, 0, 0, rve_size_x, rve_size_y)

    # Cut away fibers from the square matrix
    gmsh.model.occ.cut(
        [(2, matrix)],                              # Object to cut (matrix will get holes)
        [(2, tag) for tag in clipped_fiber_tags],   # Cutting parts (all the fibers)
        removeObject = True,                        # Delete original solid matrix
        removeTool = False)                         # Keep fibers

    # Remove clipping box
    gmsh.model.occ.remove([(2, clip_box)])

    # Apply all geometry changes to GMSH
    gmsh.model.occ.synchronize()


def extrude_to_3D(rve_size_z):
    """Extrude all 2D surfaces to 3D."""
    surfaces = gmsh.model.occ.getEntities(dim = 2)

    # Extrude all surfaces at once
    gmsh.model.occ.extrude(surfaces, 0, 0, rve_size_z)
    gmsh.model.occ.synchronize()


def assign_physical_groups(fibers, dimension, matrix_id, fiber_id, rve_size_z = None):
    """Assign material IDs based on entity size.

    Args:
        fibers: List of (x, y, radius) tuples.
        dimension: Mesh dimension (2 or 3).
        matrix_id: Material ID for the matrix.
        fiber_id: Material ID for the fibers.
        rve_size_z: RVE dimension in z-direction (only for 3D).
    """
    entities = gmsh.model.occ.getEntities(dim = dimension)

    # Entities smaller than max fiber size are fibers, rest is matrix
    max_fiber_area = max(np.pi * r ** 2 for _, _, r in fibers)
    if dimension == 3:
        max_fiber_size = max_fiber_area * rve_size_z * SIZE_TOLERANCE_FACTOR
    else:
        max_fiber_size = max_fiber_area * SIZE_TOLERANCE_FACTOR

    matrix_tags = []
    fiber_tags = []

    for dim, tag in entities:
        size = gmsh.model.occ.getMass(dim, tag)
        if size <= max_fiber_size:
            fiber_tags.append(tag)
        else:
            matrix_tags.append(tag)

    if matrix_tags:
        gmsh.model.addPhysicalGroup(dimension, matrix_tags, tag = matrix_id, name = "Matrix")
    if fiber_tags:
        gmsh.model.addPhysicalGroup(dimension, fiber_tags, tag = fiber_id, name = "Fiber")


# ================================================================
# Point Sets for Boundary Conditions
# ================================================================
def create_point_sets(dimension, mesh, rve_size_x, rve_size_y, rve_size_z = None):
    """Create point sets for RVE boundaries and corner points.
    
    Naming convention (binary: 0 = min, 1 = max):
        Boundaries 2D: X_0, X_1, Y_0, Y_1
        Boundaries 3D: + Z_0, Z_1
        Corners 2D: N_00, N_10, N_11, N_01   (x-bit, y-bit)
        Corners 3D: N_000 ... N_111          (x-bit, y-bit, z-bit)

    Args:
        dimension: Mesh dimension (2 or 3).
        mesh: Meshio mesh object.
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        rve_size_z: RVE dimension in z-direction (only for 3D).

    Returns:
        Dict mapping point set names to integer arrays.
    """
    p = mesh.points
    point_sets = {}
    
    # Boolean boundary masks
    x0 = at_min(p, 0); x1 = at_max(p, 0, rve_size_x)
    y0 = at_min(p, 1); y1 = at_max(p, 1, rve_size_y)

    # Boundaries
    point_sets["X_0"] = x0.astype(np.int32)
    point_sets["X_1"] = x1.astype(np.int32)
    point_sets["Y_0"] = y0.astype(np.int32)
    point_sets["Y_1"] = y1.astype(np.int32)

    if dimension == 2:
        point_sets["N_00"] = (x0 & y0).astype(np.int32)
        point_sets["N_10"] = (x1 & y0).astype(np.int32)
        point_sets["N_11"] = (x1 & y1).astype(np.int32)
        point_sets["N_01"] = (x0 & y1).astype(np.int32)
    else:
        z0 = at_min(p, 2); z1 = at_max(p, 2, rve_size_z)
        
        point_sets["Z_0"] = z0.astype(np.int32)
        point_sets["Z_1"] = z1.astype(np.int32)
        
        point_sets["N_000"] = (x0 & y0 & z0).astype(np.int32)
        point_sets["N_100"] = (x1 & y0 & z0).astype(np.int32)
        point_sets["N_110"] = (x1 & y1 & z0).astype(np.int32)
        point_sets["N_010"] = (x0 & y1 & z0).astype(np.int32)
        point_sets["N_001"] = (x0 & y0 & z1).astype(np.int32)
        point_sets["N_101"] = (x1 & y0 & z1).astype(np.int32)
        point_sets["N_111"] = (x1 & y1 & z1).astype(np.int32)
        point_sets["N_011"] = (x0 & y1 & z1).astype(np.int32)

    return point_sets


# ================================================================
# Periodic Node Matching for PBCs
# ================================================================
def apply_periodic_constraints(dimension, rve_size_x, rve_size_y, rve_size_z = None):
    """Apply periodic mesh constraints before meshing.

    Args:
        dimension: Mesh dimension (2 or 3).
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        rve_size_z: RVE dimension in z-direction (only for 3D).
    """
    # Entity dimension: 1 for curves (2D), 2 for surface (3D)
    entity_dim = 1 if dimension == 2 else 2
    entities = gmsh.model.occ.getEntities(dim = entity_dim)

    # Boundary entity lists
    boundaries = {
        "left": [],
        "right": [],
        "bottom": [],
        "top": [],
        "front": [],
        "back": []}

    # Classify boundary entities by position
    for dim, tag in entities:
        # Get bounding box of curve to determine its position
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.occ.getBoundingBox(dim, tag)

        if abs(x_min) < BOUNDARY_TOLERANCE and abs(x_max) < BOUNDARY_TOLERANCE:
            boundaries["left"].append(tag)
        elif abs(x_min - rve_size_x) < BOUNDARY_TOLERANCE and abs(x_max - rve_size_x) < BOUNDARY_TOLERANCE:
            boundaries["right"].append(tag)
        elif abs(y_min) < BOUNDARY_TOLERANCE and abs(y_max) < BOUNDARY_TOLERANCE:
            boundaries["bottom"].append(tag)
        elif abs(y_min - rve_size_y) < BOUNDARY_TOLERANCE and abs(y_max - rve_size_y) < BOUNDARY_TOLERANCE:
            boundaries["top"].append(tag)
        elif dimension == 3:
            if abs(z_min) < BOUNDARY_TOLERANCE and abs(z_max) < BOUNDARY_TOLERANCE:
                boundaries["front"].append(tag)
            elif abs(z_min - rve_size_z) < BOUNDARY_TOLERANCE and abs(z_max - rve_size_z) < BOUNDARY_TOLERANCE:
                boundaries["back"].append(tag)

    # Periodic pairs: (master, slave, 4x4 affine transformation matrix)
    periodic_pairs = [
        ("left", "right", 
         [1, 0, 0, rve_size_x, 
          0, 1, 0, 0,
          0, 0, 1, 0, 
          0, 0, 0, 1]),
        ("bottom", "top", 
         [1, 0, 0, 0, 
          0, 1, 0, rve_size_y, 
          0, 0, 1, 0, 
          0, 0, 0, 1])]

    if dimension == 3:
        periodic_pairs.append(("front", "back", 
                               [1, 0, 0, 0, 
                                0, 1, 0, 0, 
                                0, 0, 1, rve_size_z, 
                                0, 0, 0, 1]))

    # Match master/slave entities by bounding box in tangential directions
    for master_key, slave_key, transform in periodic_pairs:
        for master_tag in boundaries[master_key]:
            master_bounding_box = gmsh.model.occ.getBoundingBox(entity_dim, master_tag)

            for slave_tag in boundaries[slave_key]:
                slave_bounding_box = gmsh.model.occ.getBoundingBox(entity_dim, slave_tag)

                # Check if bounding boxes match in tangential directions
                # (i.e., all coordinates except the periodic direction must match)
                if dimension == 2:
                    if master_key == "left":
                        match = (abs(master_bounding_box[1] - slave_bounding_box[1]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[4] - slave_bounding_box[4]) < BOUNDARY_TOLERANCE)
                    else:  # bottom
                        match = (abs(master_bounding_box[0] - slave_bounding_box[0]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[3] - slave_bounding_box[3]) < BOUNDARY_TOLERANCE)
                else:  # 3D
                    if master_key == "left":
                        match = (abs(master_bounding_box[1] - slave_bounding_box[1]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[4] - slave_bounding_box[4]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[2] - slave_bounding_box[2]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[5] - slave_bounding_box[5]) < BOUNDARY_TOLERANCE)
                    elif master_key == "bottom":
                        match = (abs(master_bounding_box[0] - slave_bounding_box[0]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[3] - slave_bounding_box[3]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[2] - slave_bounding_box[2]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[5] - slave_bounding_box[5]) < BOUNDARY_TOLERANCE)
                    else:  # front
                        match = (abs(master_bounding_box[0] - slave_bounding_box[0]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[3] - slave_bounding_box[3]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[1] - slave_bounding_box[1]) < BOUNDARY_TOLERANCE
                                 and abs(master_bounding_box[4] - slave_bounding_box[4]) < BOUNDARY_TOLERANCE)

                if match:
                    gmsh.model.mesh.setPeriodic(entity_dim, [slave_tag], [master_tag], transform)
                    break


# ================================================================
# Mesh Generation and Export
# ================================================================
def generate_mesh(dimension, mesh_size_min, mesh_size_max, mesh_element_order):
    """Configure and generate the GMSH mesh."""
    # Element size control
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)

    # Element order (1 = linear, 2 = quadratic)
    gmsh.option.setNumber("Mesh.ElementOrder", mesh_element_order)

    gmsh.model.mesh.generate(dimension)


def convert_to_vtu(msh_path, vtu_path, dimension, rve_size_x, rve_size_y, rve_size_z = None):
    """Convert MSH to 4C-compatible VTU format.

    Args:
        msh_path: Path to the input MSH file.
        vtu_path: Path for the output VTU file.
        dimension: Mesh dimension (2 or 3).
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        rve_size_z: RVE dimension in z-direction (only for 3D).

    Returns:
        Meshio mesh object.
    """
    mesh = meshio.read(msh_path)

    # Clean up unnecessary data
    mesh.cell_sets = {}
    mesh.point_data = {}

    # Rename gmsh:physical to block_id for 4C
    if "gmsh:physical" in mesh.cell_data:
        mesh.cell_data["block_id"] = mesh.cell_data.pop("gmsh:physical")

    # Remove other gmsh fields
    for key in list(mesh.cell_data.keys()):
        if key.startswith("gmsh:") or key.startswith("cell_tags"):
            del mesh.cell_data[key]

    # Add point sets for boundary conditions
    point_sets = create_point_sets(dimension, mesh, rve_size_x, rve_size_y, rve_size_z)

    # Add point sets to mesh for boundary condition assignment in 4C
    for name, data in point_sets.items():
        mesh.point_data[name] = data #todo

    # Write VTU file
    meshio.write(vtu_path, mesh, binary = True)

    return mesh


# ================================================================
# Main Processing Function
# ================================================================
def process_csv_to_vtu(csv_path, dimension, rve_size_x, rve_size_y, mesh_size_min, mesh_size_max,
                       mesh_element_order, matrix_id, fiber_id, msh_dir, vtu_dir, rve_size_z = None):
    """Process single CSV file to generate a 4C-compatible VTU mesh.

    Args:
        csv_path: Path to the fiber distribution CSV file.
        dimension: Mesh dimension (2 or 3).
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        rve_size_z: RVE dimension in z-direction (only for 3D).
        mesh_size_min: Minimum element size.
        mesh_size_max: Maximum element size.
        mesh_element_order: Element order (1=linear, 2=quadratic).
        matrix_id: Material ID for the matrix.
        fiber_id: Material ID for the fibers.
        msh_dir: Directory for MSH output.
        vtu_dir: Directory for VTU output.

    Returns:
        Tuple of (n_fibers, n_nodes, n_elements).
    """
    csv_path = Path(csv_path)
    msh_dir = Path(msh_dir)
    vtu_dir = Path(vtu_dir)
    msh_dir.mkdir(parents = True, exist_ok = True)
    vtu_dir.mkdir(parents = True, exist_ok = True)

    base_name = csv_path.stem
    msh_path = msh_dir / f"{base_name}.msh"
    vtu_path = vtu_dir / f"{base_name}.vtu"

    fibers = load_fiber_data(csv_path)
    n_fibers = len(fibers)

    gmsh.initialize(interruptible = False)
    gmsh.option.setNumber("General.Terminal", 0)  # GMSH console output (0 = silent, 1 = detailed)
    # Force single-threaded meshing for deterministic node numbering across runs/machines.
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add("RVE")

    create_2D_geometry(fibers, rve_size_x, rve_size_y)

    if dimension == 2:
        apply_periodic_constraints(dimension, rve_size_x, rve_size_y)
    elif dimension == 3:
        extrude_to_3D(rve_size_z)
        apply_periodic_constraints(dimension, rve_size_x, rve_size_y, rve_size_z)

    assign_physical_groups(fibers, dimension, matrix_id, fiber_id, rve_size_z)
    generate_mesh(dimension, mesh_size_min, mesh_size_max, mesh_element_order)

    # Export mesh in MSH format
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    mesh = convert_to_vtu(msh_path, vtu_path, dimension, rve_size_x, rve_size_y, rve_size_z)

    n_nodes = len(mesh.points)
    n_elements = sum(len(cells.data) for cells in mesh.cells)

    return n_fibers, n_nodes, n_elements


# ================================================================
# Main
# ================================================================
def main():
    """Run mesh generation from config files."""
    from config_loader import config
    from plot_vtu_mesh import plot_single_mesh

    print("=" * 60)
    print("VTU MESH GENERATOR FOR 4C")
    print("=" * 60)

    dimension = config.get("rve.rve.dimension")
    rve_size_x = config.get("rve.rve.size_x")
    rve_size_y = config.get("rve.rve.size_y")
    rve_size_z = config.get("rve.rve.size_z") if dimension == 3 else None
    mesh_size_min = config.get("rve.mesh.size_min")
    mesh_size_max = config.get("rve.mesh.size_max")
    mesh_element_order = config.get("rve.mesh.element_order")
    matrix_id = config.get("rve.materials.matrix_id")
    fiber_id = config.get("rve.materials.fiber_id")
    csv_dir = Path(config.get("paths.data.csv"))
    msh_dir = config.get("paths.data.msh")
    vtu_dir = config.get("paths.data.vtu")
    enable_plotting = True
    plot_frequency = 1

    print(f"{'Dimension:':<30} {dimension}D")
    if dimension == 3:
        print(f"{'RVE Size (X x Y x Z):':<30} {rve_size_x} x {rve_size_y} x {rve_size_z}")
    else:
        print(f"{'RVE Size (X x Y):':<30} {rve_size_x} x {rve_size_y}")
    print(f"{'Min. Mesh Size:':<30} {mesh_size_min}")
    print(f"{'Max. Mesh Size:':<30} {mesh_size_max}")
    print(f"{'Mesh Element Order:':<30} {mesh_element_order}\n")

    # Common args for process_csv_to_vtu
    mesh_args = dict(dimension = dimension,
                     rve_size_x = rve_size_x,
                     rve_size_y = rve_size_y,
                     rve_size_z = rve_size_z,
                     mesh_size_min = mesh_size_min,
                     mesh_size_max = mesh_size_max,
                     mesh_element_order = mesh_element_order,
                     matrix_id = matrix_id,
                     fiber_id = fiber_id,
                     msh_dir = msh_dir,
                     vtu_dir = vtu_dir)

    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV file(s)")

    for i, csv_path in enumerate(csv_files, 1):
        try:
            n_fibers, n_nodes, n_elements = process_csv_to_vtu(csv_path, **mesh_args)
            print(f"{csv_path.name}: {n_fibers} fibers, {n_nodes} nodes, {n_elements} elements")

            if enable_plotting and i % plot_frequency == 0:
                vtu_path = Path(vtu_dir) / f"{csv_path.stem}.vtu"
                plot_single_mesh(vtu_path)

        except Exception as e:
            print(f"{csv_path.name}: Error - {e}")

    print("\n" + "-" * 60)
    print("MESH GENERATION COMPLETE")
    print("-" * 60)


if __name__ == "__main__":
    main()
