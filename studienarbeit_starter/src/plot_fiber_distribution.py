"""Visualization of RSE and RFR fiber distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm

from config_loader import config
from plot_utils import apply_square_ticks

# ================================================================
# Plot Settings
# ================================================================
FIGURE_SIZE = (12, 6)
FONT_FAMILY = "DejaVu Sans Mono"
FONT_SIZE = 14
FONT_WEIGHT = "normal"

# Axes settings
SHOW_TICKS = True
TITLE_PADDING = 10

# Grid settings
SHOW_GRID = False
GRID_COLOR = "#000000"
GRID_LINESTYLE = "-"
GRID_LINEWIDTH = 0.5
GRID_ALPHA = 1.0

# Fiber colors
FIBER_COLORS = {"normal": "black",
                "boundary": "#ED6E00",  # UniBw Orange
                "periodic": "#646363",  # UniBw Grey
                "rfr": "black"}

# Output settings
FILE_FORMATS = ["pdf"]
DPI = 300
SAVE_PLOTS = True


# ================================================================
# Plotting Function
# ================================================================
def plot_comparison(before_centers, before_types, before_radii, after_centers, after_radii,
                    rve_size_x, rve_size_y, identifier, vf_percent, output_dir = None, show = True):
    """Visualize fibers before (RSE) and after (RFR) removal.

    Args:
        before_centers: Fiber positions before RFR.
        before_types: Fiber type strings ('normal', 'boundary', 'periodic').
        after_centers: Fiber positions after RFR.
        fiber_radius: Radius of each fiber.
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        identifier: String identifier for filename.
        vf_percent: Target volume fraction in percent.
    """
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["font.weight"] = FONT_WEIGHT

    fig, axs = plt.subplots(1, 2, figsize = FIGURE_SIZE)

    # Left subplot: RSE Variant (before RFR)
    axs[0].set_title("RSE Variant", pad = TITLE_PADDING, fontsize = FONT_SIZE)

    for fiber, fiber_type, fiber_radius in zip(before_centers, before_types, before_radii):
        display_color = FIBER_COLORS[fiber_type]
        axs[0].add_artist(plt.Circle((fiber[0], fiber[1]), fiber_radius, color = display_color, linewidth = 0))

    axs[0].set_xlim([0, rve_size_x])
    axs[0].set_ylim([0, rve_size_y])
    axs[0].set_aspect("equal")

    # Right subplot: RFR Variant (after RFR)
    axs[1].set_title(rf"RFR Variant $\left(V_{{F}} = {vf_percent:.0f}\%\right)$", pad = TITLE_PADDING, fontsize = FONT_SIZE)

    for fiber, fiber_radius in zip(after_centers, after_radii):
        axs[1].add_artist(plt.Circle((fiber[0], fiber[1]), fiber_radius, color = FIBER_COLORS["rfr"], linewidth = 0))

    axs[1].set_xlim([0, rve_size_x])
    axs[1].set_ylim([0, rve_size_y])
    axs[1].set_aspect("equal")

    # Apply settings to both subplots
    for ax in axs:
        if SHOW_TICKS:
            apply_square_ticks(ax, rve_size_x, rve_size_y)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if SHOW_GRID:
            ax.grid(True, which = "major", color = GRID_COLOR, linestyle = GRID_LINESTYLE, linewidth = GRID_LINEWIDTH, alpha = GRID_ALPHA)
            ax.minorticks_on()
            ax.grid(True, which = "minor", color = GRID_COLOR, linestyle = GRID_LINESTYLE, linewidth = GRID_LINEWIDTH * 0.5, alpha = GRID_ALPHA * 0.5)

    plt.tight_layout()

    # Save plot in specified formats
    if SAVE_PLOTS:
        if output_dir is None:
            output_dir = Path(config.get("paths.data.images.fiber_distribution"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)

        for fmt in FILE_FORMATS:
            file_name = f"{config.get('paths.file_naming.image_rse_rfr')}_{identifier.split('_')[-1]}.{fmt}"
            file_path = output_dir / file_name
            plt.savefig(file_path, dpi = DPI, bbox_inches = "tight")

    # Show plots if enabled
    if show:
        plt.show(block = False)
        plt.pause(0.01)  # Small pause to render the plot

    plt.close()


# ================================================================
# Standalone re-plot from on-disk job data
# ================================================================
def _load_fiber_data(job_dir):
    """Load before/after-RFR fiber data + parameters from a single job directory.

    Expects in ``job_dir``:
        rve_rse_<id>.csv  — columns: x_coordinate, y_coordinate, radius, type
        rve_rfr_<id>.csv  — columns: x_coordinate, y_coordinate, radius
        metadata.yaml     — inputs: vf_percent, fiber_radius, rve_size

    Returns:
        Kwargs dict for ``plot_comparison`` if all files exist, else None.
    """
    job_dir = Path(job_dir)
    job_id = int(job_dir.name)
    prefix = config.get("paths.file_naming.fiber_distribution")

    rse_csv = job_dir / f"{prefix}_rse_{job_id:04d}.csv"
    rfr_csv = job_dir / f"{prefix}_rfr_{job_id:04d}.csv"
    metadata_path = job_dir / "metadata.yaml"

    if not (rse_csv.exists() and rfr_csv.exists() and metadata_path.exists()):
        return None

    rse_df = pd.read_csv(rse_csv)
    rfr_df = pd.read_csv(rfr_csv)
    inputs = yaml.safe_load(metadata_path.read_text()).get("inputs", {})

    rve_size = inputs.get("rve_size")
    return {
        "before_centers": rse_df[["x_coordinate", "y_coordinate"]].values.tolist(),
        "before_types":   rse_df["type"].tolist(),
        "before_radii":   rse_df["radius"].tolist(),
        "after_centers":  rfr_df[["x_coordinate", "y_coordinate"]].values.tolist(),
        "after_radii":    rfr_df["radius"].tolist(),
        "rve_size_x":     rve_size,
        "rve_size_y":     rve_size,
        "identifier":     f"{prefix}_rfr_{job_id:04d}",
        "vf_percent":     inputs.get("vf_percent"),
    }


def main():
    """Re-plot fiber distribution for every job of the current QUEENS experiment."""
    experiment_name = config.get("queens.name")
    experiment_dir  = Path(config.get("paths.base.data")) / experiment_name

    job_dirs = sorted(
        [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key = lambda d: int(d.name),
    )
    if not job_dirs:
        print(f"No jobs found in {experiment_dir}")
        return

    print(f"Plotting fiber distributions for {len(job_dirs)} job(s)...")
    for jd in tqdm(job_dirs, desc = "Fiber distribution", unit = "job"):
        data = _load_fiber_data(jd)
        if data is None:
            tqdm.write(f"  [skip] {jd.name}: missing CSV(s) or metadata.yaml")
            continue
        plot_comparison(**data, output_dir = jd, show = False)


if __name__ == "__main__":
    main()
