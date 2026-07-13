"""VTU mesh visualization with matplotlib."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from config_loader import config
from plot_utils import apply_square_ticks

# ================================================================
# Plot Settings
# ================================================================
FIGURE_SIZE = (6, 6)
FONT_FAMILY = "DejaVu Sans Mono"
FONT_SIZE = 14
FONT_WEIGHT = "normal"

# Axes settings
SHOW_TICKS = True
TITLE_PADDING = 10

# Mesh settings
MESH_LINEWIDTH = 0.5
COLORS = {"matrix": "#646363",  # UniBw Grey
          "fiber": "#ED6E00",  # UniBw Orange
          "mesh": "black"}  

# Output settings
FILE_FORMATS = ["pdf"]
DPI = 300
SAVE_PLOTS = True


# ================================================================
# Plotting Function
# ================================================================
def plot_single_mesh(vtu_path, vf_percent = None, title = None, output_dir = None, show = True):
    """Plot a single VTU mesh with matrix/fiber coloring.

    Args:
        vtu_path: Path to the VTU file.
        vf_percent: Fiber volume fraction in percent. Used for title.
        title: Optional plot title. Overrides vf_percent-based title.
    """
    vtu_path = Path(vtu_path)
    mesh = meshio.read(vtu_path)

    matrix_id = config.get("rve.materials.matrix_id")
    fiber_id = config.get("rve.materials.fiber_id")

    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["font.weight"] = FONT_WEIGHT

    fig, ax = plt.subplots(1, 1, figsize = FIGURE_SIZE)

    # Get triangle connectivity, points, and block IDs
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    block_ids = mesh.cell_data["block_id"][0]

    # Plot matrix and fiber triangles separately
    for block_id, color_key in [(matrix_id, "matrix"), (fiber_id, "fiber")]:
        mask = block_ids == block_id
        cell_indices = triangles[mask]
        triangulation = tri.Triangulation(points[:, 0], points[:, 1], cell_indices)
        # Single-color colormap to force tripcolor to use our color
        cmap = ListedColormap([COLORS[color_key]])
        ax.tripcolor(triangulation, facecolors = np.ones(mask.sum()), cmap = cmap, edgecolors = COLORS["mesh"], linewidth = MESH_LINEWIDTH)

    ax.set_xlim([points[:, 0].min(), points[:, 0].max()])
    ax.set_ylim([points[:, 1].min(), points[:, 1].max()])
    ax.set_aspect("equal")

    if SHOW_TICKS:
        apply_square_ticks(ax, points[:, 0].max(), points[:, 1].max())
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is None:
        if vf_percent is not None:
            title = rf"RVE Mesh $\left(V_{{F}} = {vf_percent:.0f}\%\right)$"
        else:
            match = re.search(r"vf(\d+)", vtu_path.stem)
            title = rf"RVE Mesh $\left(V_{{F}} = {int(match.group(1))}\%\right)$" if match else vtu_path.stem

    ax.set_title(title, pad = TITLE_PADDING, fontsize = FONT_SIZE)

    plt.tight_layout()

    if SAVE_PLOTS:
        if output_dir is None:
            output_dir = Path(config.get("paths.data.images.mesh"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)

        for fmt in FILE_FORMATS:
            # Identifier = trailing job_id (last underscore segment), robust to "rve_<id>" and "rve_rfr_<id>"
            identifier = vtu_path.stem.split("_")[-1]
            file_path = output_dir / f"{config.get('paths.file_naming.image_mesh')}_{identifier}.{fmt}"
            plt.savefig(file_path, dpi = DPI)

    if show:
        plt.show(block = False)
        plt.pause(0.01)

    plt.close()


# ================================================================
# Main
# ================================================================
def main():
    """Re-plot meshes for every job of the current QUEENS experiment."""
    experiment_name = config.get("queens.name")
    experiment_dir  = Path(config.get("paths.base.data")) / experiment_name

    job_dirs = sorted(
        [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key = lambda d: int(d.name),
    )
    if not job_dirs:
        print(f"No jobs found in {experiment_dir}")
        return

    print(f"Plotting meshes for {len(job_dirs)} job(s)...")
    for jd in tqdm(job_dirs, desc = "Mesh", unit = "job"):
        vtu_files = sorted(jd.glob("*.vtu"))
        if not vtu_files:
            tqdm.write(f"  [skip] {jd.name}: no VTU")
            continue

        vf_percent = None
        metadata_path = jd / "metadata.yaml"
        if metadata_path.exists():
            vf_percent = yaml.safe_load(metadata_path.read_text()).get("inputs", {}).get("vf_percent")

        # One VTU per job in experiment mode (the RFR mesh)
        plot_single_mesh(vtu_files[0], vf_percent = vf_percent, output_dir = jd, show = False)


if __name__ == "__main__":
    main()
