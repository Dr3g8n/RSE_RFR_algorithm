"""Configuration loader for RVE fiber distribution projects."""

import json
import math
from pathlib import Path


class ConfigLoader:
    """Load JSON config files and calculate dynamic values."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_dir = self.project_root / "config"

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory '{self.config_dir}' not found")

        self.config = {}
        self.load_all_configs()

    def load_json(self, filename):
        """Load a single JSON configuration file."""
        filepath = self.config_dir / filename
        return json.loads(filepath.read_text(encoding = "utf-8"))

    def load_all_configs(self):
        """Load all configuration files and calculate dynamic values."""
        self.config["rve"] = self.load_json("rve.json")
        self.config["algorithm"] = self.load_json("algorithm.json")
        self.config["paths"] = self._resolve_paths(self.load_json("paths.json"))
        self.config["targets"] = self.load_json("target_vfs.json")
        self.config["queens"] = self.load_json("queens.json")
        self._calculate_dynamic_values()

    def _resolve_paths(self, obj):
        """Recursively convert relative paths (./) to absolute paths."""
        if isinstance(obj, str) and obj.startswith("./"):
            return str(self.project_root / obj[2:])
        elif isinstance(obj, dict):
            return {k: self._resolve_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_paths(item) for item in obj]
        return obj

    def _calculate_dynamic_values(self):
        """Calculate values that depend on rve configuration."""
        size_x = self.config["rve"]["rve"]["size_x"]
        size_y = self.config["rve"]["rve"]["size_y"]

        # Calculate periodic boundary shifts for 2D periodicity (8 directions)
        # Creates copies of fibers at boundaries to ensure periodic boundary conditions
        self.config["algorithm"]["periodic_shifts"] = [
            [-size_x, 0],  # Left
            [size_x, 0],  # Right
            [0, -size_y],  # Bottom
            [0, size_y],  # Top
            [-size_x, -size_y],  # Bottom-Left
            [-size_x, size_y],  # Top-Left
            [size_x, -size_y],  # Bottom-Right
            [size_x, size_y],  # Top-Right
        ]

        # Use the larger radius (normal) vs. with CZM layer)
        fiber_radius_max = self.config["rve"]["fiber"]["radius_max"]
        fiber_radius_with_czm = self.config["rve"]["fiber"]["radius_with_czm"]
        safety_radius = max(fiber_radius_max, fiber_radius_with_czm)
        self.config["rve"]["fiber"]["radius"] = safety_radius

        # Buffer zone for periodic boundary fiber copies
        multiplier = self.config["algorithm"]["periodic_boundary_buffer_multiplier"]
        self.config["rve"]["rve"]["periodic_boundary_buffer"] = multiplier * safety_radius

        # Dynamic max fiber count (safety cap for RSE loop)
        self.config["algorithm"]["max_fibers_count"] = int(1.5 * (size_x * size_y) / (math.pi * safety_radius**2))
        
        # Resolve solver name → ID
        solver_name = self.config["rve"]["solver"]["name"]
        solver_mapping = self.config["rve"]["solver"]["mapping"]
        self.config["rve"]["solver"]["id"] = solver_mapping[solver_name]
        
    def get_exclusion_zones(self, fiber_radius, size_x, size_y):
        """Compute boundary exclusion zones for a given fiber radius and RVE size.

        Prevents slivers thinner than mesh_size_min at RVE boundaries.
        A fiber center must not land in the band (r - mesh_min, r + mesh_min)
        from any wall, ensuring at least one mesh cell fits in every gap or sliver.

        Args:
            fiber_radius: Fiber radius for the current job (QUEENS parameter).
            size_x:       RVE size in x for the current job (QUEENS parameter).
            size_y:       RVE size in y for the current job (QUEENS parameter).

        Returns:
            List of exclusion zone dicts with x_range and y_range.
        """
        mesh_min = self.config["rve"]["mesh"]["size_min"]
        d_inner = fiber_radius - mesh_min
        d_outer = fiber_radius + mesh_min
        buffer = fiber_radius * self.config["algorithm"]["periodic_boundary_buffer_multiplier"]

        return [
            # Inner exclusion zones (inside RVE, near walls)
            {"x_range": [d_inner, d_outer],                    "y_range": [0, size_y]},
            {"x_range": [size_x - d_outer, size_x - d_inner],  "y_range": [0, size_y]},
            {"x_range": [0, size_x],                           "y_range": [d_inner, d_outer]},
            {"x_range": [0, size_x],                           "y_range": [size_y - d_outer, size_y - d_inner]},
            # Outer exclusion zones (buffer zone outside RVE, near walls)
            {"x_range": [-buffer, -buffer + mesh_min],                  "y_range": [-buffer, size_y + buffer]},
            {"x_range": [size_x + buffer - mesh_min, size_x + buffer],  "y_range": [-buffer, size_y + buffer]},
            {"x_range": [-buffer, size_x + buffer],                     "y_range": [-buffer, -buffer + mesh_min]},
            {"x_range": [-buffer, size_x + buffer],                     "y_range": [size_y + buffer - mesh_min, size_y + buffer]},
        ]
    
    def build_queens_parameters(self):
        """Build QUEENS Parameters from queens.parameters config."""
        from queens.parameters import Parameters

        param_config = self.get("queens.parameters")
        return Parameters(**{name: self._build_distribution(p) for name, p in param_config.items()})

    @staticmethod
    def _build_distribution(p):
        """Build a single QUEENS distribution from a parameter config dict."""
        from queens.distributions import TruncatedNormal, Uniform

        dtype = p["type"]
        if dtype == "uniform":
            return Uniform(lower_bound = p["lower"], upper_bound = p["upper"])
        if dtype == "truncated_normal":
            return TruncatedNormal(
                unbounded_mean = p["unbounded_mean"],
                unbounded_std  = p["unbounded_std"],
                lower_bound    = p["lower"],
                upper_bound    = p["upper"],
            )
        if dtype == "fixed":
            # Degenerate Uniform: tiny eps so QUEENS' lower < upper check passes;
            # Grid with num_grid_points=1 returns linspace(v, v+eps, 1) = [v].
            v = float(p["value"])
            eps = max(abs(v), 1.0) * 1e-12
            return Uniform(lower_bound = v, upper_bound = v + eps)
        raise ValueError(f"Unknown distribution type: '{dtype}'")

    def build_queens_iterator(self, model, parameters, global_settings, result_description):
        """Build a QUEENS Iterator from queens.iterator config (seed from algorithm.seed)."""
        from queens.iterators.grid import Grid
        from queens.iterators.latin_hypercube_sampling import LatinHypercubeSampling
        from queens.iterators.monte_carlo import MonteCarlo
        from queens.iterators.sobol_index import SobolIndex

        iter_type = self.get("queens.iterator.type")
        iter_cfg  = self.get(f"queens.iterator.{iter_type}")
        seed      = self.get("algorithm.seed")

        common = dict(
            model              = model,
            parameters         = parameters,
            global_settings    = global_settings,
            result_description = result_description,
        )

        if iter_type == "grid":
            grid_points_map = iter_cfg.get("grid_points", {})
            param_names = list(self.get("queens.parameters").keys())
            grid_design = {
                name: {
                    "num_grid_points": grid_points_map.get(name, 1),
                    "axis_type":       "lin",
                    "data_type":       "FLOAT",
                }
                for name in param_names
            }
            return Grid(**common, grid_design = grid_design)

        if iter_type == "monte_carlo":
            return MonteCarlo(
                **common,
                seed        = seed,
                num_samples = iter_cfg["num_samples"],
            )

        if iter_type == "latin_hypercube_sampling":
            return LatinHypercubeSampling(
                **common,
                seed        = seed,
                num_samples = iter_cfg["num_samples"],
            )

        if iter_type == "sobol":
            return SobolIndex(
                **common,
                seed                  = seed,
                num_samples           = iter_cfg["num_samples"],
                calc_second_order     = iter_cfg["calc_second_order"],
                num_bootstrap_samples = iter_cfg["num_bootstrap_samples"],
                confidence_level      = iter_cfg["confidence_level"],
            )

        raise ValueError(f"Unknown iterator type: '{iter_type}'")
        
    def get(self, key_path, default = None):
        """Access configuration using dot notation.

        Args:
            key_path: Dot-separated key (e.g. 'rve.rve.size_x').
            default: Fallback value if key not found.

        Returns:
            The configuration value.

        Raises:
            KeyError: If key not found and no default provided.
        """
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key '{key_path}' not found")

    def get_all(self):
        """Return the entire configuration dictionary."""
        return self.config

    def print_summary(self):
        """Print a formatted summary of the configuration."""
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        print(f"{'RVE Size (X x Y x Z):':<30} {self.get('rve.rve.size_x')} x {self.get('rve.rve.size_y')} x {self.get('rve.rve.size_z')}")
        print(f"{'Fiber Radius:':<30} {self.get('rve.fiber.radius')}")
        print(f"{'Min. Mesh Size:':<30} {self.get('rve.mesh.size_min')}")
        print(f"{'Max. Mesh Size:':<30} {self.get('rve.mesh.size_max')}")
        print(f"{'Mesh Element Order:':<30} {self.get('rve.mesh.element_order')}\n")

        use_specific = self.get("targets.use_specific_targets", False)
        if use_specific:
            configs = self.get("targets.specific_target_configurations")
            print("-" * 60)
            print(f"Mode: SPECIFIC TARGETS ({len(configs)} configurations)")
            for i, cfg in enumerate(configs, 1):
                print(f"  {i:2d}.  VF: {cfg['vf_percent']:2d}%, Targets: {len(cfg['targets'])}")
            print("-" * 60)
        else:
            targets = self.get("targets.volume_fraction_combinations")
            print("-" * 60)
            print(f"Mode: CLASSICAL ({len(targets)} volume fraction(s))")
            for i, vf in enumerate(targets, 1):
                print(f"  {i:2d}.  FVF: {vf['vf_percent']:2d}%,  RSE: {vf['RSE_variants']:3d},  RFR: {vf['RFR_variants']:3d}")
            print("-" * 60)

# Global config instance
config = ConfigLoader()

if __name__ == "__main__":
    config.print_summary()
