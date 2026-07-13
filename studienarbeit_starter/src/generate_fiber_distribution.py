"""Fiber distribution generation using RSE and RFR algorithms."""

import csv
from math import cos, pi, sin
from pathlib import Path

import numpy as np
from tqdm import tqdm

from seeding import derive_job_rng

# ================================================================
# Helper Functions
# ================================================================
def check_within_tolerance(coordinate, upper_limit, lower_limit, exclusion_zones):
    """Check if fiber coordinate is within RVE boundaries and outside exclusion zones.

    Args:
        coordinate: [x, y] position of the fiber center.
        upper_limit: Upper boundary limit (with tolerance).
        lower_limit: Lower boundary limit (with tolerance).
        exclusion_zones: List of exclusion zone dicts with x_range and y_range.

    Returns:
        True if coordinate is valid, False otherwise.
    """
    x, y = coordinate

    if x < lower_limit or x > upper_limit or y < lower_limit or y > upper_limit:
        return False

    for zone in exclusion_zones:
        x_range = zone["x_range"]
        y_range = zone["y_range"]
        if (x_range[0] < x < x_range[1]) and (y_range[0] < y < y_range[1]):
            return False

    return True


def is_fiber_at_boundary(coordinate, fiber_radius, rve_size_x, rve_size_y):
    """Check if a fiber touches or crosses an RVE boundary."""
    x, y = coordinate

    return (x - fiber_radius < 0 or x + fiber_radius > rve_size_x or 
            y - fiber_radius < 0 or y + fiber_radius > rve_size_y)


def generate_periodic_fibers(coordinate, fiber_radius, centers, periodic_shifts, upper_limit, lower_limit):
    """Generate periodic copies of a boundary fiber.

    Args:
        coordinate: [x, y] position of the original fiber.
        fiber_radius: Radius of the fiber.
        centers: List of existing fiber center positions.
        periodic_shifts: List of [dx, dy] shift vectors for periodicity.
        upper_limit: Upper boundary limit (with tolerance).
        lower_limit: Lower boundary limit (with tolerance).

    Returns:
        List of new periodic fiber positions, or empty list if overlap detected.
    """
    new_fibers = []

    for shift in periodic_shifts:
        shifted_coordinates = [coordinate[0] + shift[0], coordinate[1] + shift[1]]

        # Check if shifted fiber is within tolerance and doesn't overlap with other fibers
        if check_within_tolerance(shifted_coordinates, upper_limit, lower_limit, []):
            if all(np.linalg.norm(np.array(shifted_coordinates) - np.array(existing_fiber)) >= 2 * fiber_radius for existing_fiber in centers):
                new_fibers.append(shifted_coordinates)
            else:
                return []  # Overlap detected, abort periodic generation

    return new_fibers


def remove_fiber_with_periodicity(coordinates, fiber_centers, periodic_shifts):
    """Remove a fiber and all its periodic copies.

    Args:
        coordinates: [x, y] position of the fiber to remove.
        fiber_centers: List of all fiber center positions.
        periodic_shifts: List of [dx, dy] shift vectors.

    Returns:
        Updated fiber_centers list with fiber and copies removed.
    """
    fibers_to_remove = [coordinates]

    for shift in periodic_shifts:
        fibers_to_remove.append([coordinates[0] + shift[0], coordinates[1] + shift[1]])

    # Remove all copies from fiber list
    fiber_centers = [f for f in fiber_centers if all(not np.allclose(f, remove_fiber, atol = 1e-6) for remove_fiber in fibers_to_remove)]

    return fiber_centers


def calculate_fiber_volume_fraction(fiber_count, fiber_radius, rve_size_x, rve_size_y):
    """Calculate the actual fiber volume fraction."""
    fiber_area = np.pi * (fiber_radius ** 2)
    total_fiber_area = fiber_count * fiber_area
    rve_area = rve_size_x * rve_size_y

    return total_fiber_area / rve_area


def count_original_fibers(fiber_centers, fiber_types):
    """Count only original fibers (exclude periodic copies)."""
    return sum(1 for ft in fiber_types if ft != "periodic")


def calculate_target_fiber_count(volume_fraction, fiber_radius, rve_size_x, rve_size_y):
    """Calculate number of fibers needed for a target volume fraction."""
    fiber_area = np.pi * (fiber_radius ** 2)
    rve_area = rve_size_x * rve_size_y

    return int((volume_fraction * rve_area) / fiber_area)


# ================================================================
# Input/Output Functions
# ================================================================
def save_to_csv(fiber_centers, fiber_radius, csv_dir, identifier, fiber_types = None):
    """Save fiber coordinates to CSV file.

    If ``fiber_types`` is given, an extra ``type`` column is written
    (used for the RSE CSV which carries normal/boundary/periodic markers).
    """
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents = True, exist_ok = True)

    csv_path = csv_dir / f"{identifier}.csv"
    has_types = fiber_types is not None

    with open(csv_path, mode = "w", newline = "") as file:
        writer = csv.writer(file)
        header = ["x_coordinate", "y_coordinate", "radius"]
        if has_types:
            header.append("type")
        writer.writerow(header)
        for i, fiber in enumerate(fiber_centers):
            row = [fiber[0], fiber[1], fiber_radius]
            if has_types:
                row.append(fiber_types[i])
            writer.writerow(row)


# ================================================================
# Core Algorithms
# ================================================================
def RSE_algorithm(rve_size_x, rve_size_y,  fiber_radius, exclusion_zones, max_fibers_count, max_failed_attempts, 
                  max_trials_per_fiber, distance_factors, periodic_boundary_buffer, periodic_shifts, rng):
    """Generate initial fiber distribution using Random Sequential Expansion (RSE).

    Args:
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        fiber_radius: Radius of each fiber.
        exclusion_zones: List of exclusion zone dicts.
        max_fibers_count: Maximum number of fibers to place.
        max_failed_attempts: Stop after this many consecutive failed placements.
        max_trials_per_fiber: Maximum placement attempts per fiber.
        distance_factors: Dict with 'min_distance' and 'max_distance' factors.
        periodic_boundary_buffer: Buffer zone size beyond RVE boundaries.
        periodic_shifts: List of [dx, dy] shift vectors for periodicity.
        rng: random.Random instance for isolated, reproducible randomness.

    Returns:
        Tuple of (fiber_centers, fiber_types, fiber_count,
        virtual_fiber_count, original_fibers).
    """
    lower_limit = 0
    tolerance_lower = lower_limit - periodic_boundary_buffer
    tolerance_upper = max(rve_size_x, rve_size_y) + periodic_boundary_buffer

    # Calculate distance range for new fibers
    min_distance = distance_factors["min_distance"] * fiber_radius + 2 * fiber_radius
    max_distance = distance_factors["max_distance"] * fiber_radius + 2 * fiber_radius

    fiber_centers = []
    fiber_types = []  # 'normal' = interior fiber, 'boundary' = at edge, 'periodic' = copy

    # Place first fiber randomly
    first_coordinate = [rng.uniform(lower_limit, rve_size_x), rng.uniform(lower_limit, rve_size_y)]
    fiber_centers.append(first_coordinate)
    
    if is_fiber_at_boundary(first_coordinate, fiber_radius, rve_size_x, rve_size_y):
        new_fibers = generate_periodic_fibers(first_coordinate, fiber_radius, fiber_centers,
                                              periodic_shifts, tolerance_upper, tolerance_lower)
        
        fiber_types.append("boundary")
        for fiber in new_fibers:
            fiber_centers.append(fiber)
            fiber_types.append("periodic")
    else:
        fiber_types.append("normal")
    
    fiber_count = 1
    virtual_fiber_count = len(fiber_centers)
    fail_count = 0

    # Main RSE loop
    while fail_count < max_failed_attempts and fiber_count < max_fibers_count:
        # Select random existing fiber as seed
        target_fiber = fiber_centers[rng.randint(0, virtual_fiber_count - 1)]

        trial = 0
        added_fiber = False

        while trial < max_trials_per_fiber:
            # Generate random position around target fiber
            theta = rng.uniform(0, 2 * pi)
            distance = rng.uniform(min_distance, max_distance)
            new_coordinate = [target_fiber[0] + distance * cos(theta), target_fiber[1] + distance * sin(theta)]

            if not check_within_tolerance(new_coordinate, tolerance_upper, tolerance_lower, exclusion_zones):
                trial += 1
                continue

            # Check for overlaps with existing fibers
            if all(np.linalg.norm(np.array(new_coordinate) - np.array(existing_fiber)) >= 2 * fiber_radius for existing_fiber in fiber_centers):

                if is_fiber_at_boundary(new_coordinate, fiber_radius, rve_size_x, rve_size_y):
                    new_fibers = generate_periodic_fibers(new_coordinate, fiber_radius, fiber_centers, periodic_shifts, 
                                                          tolerance_upper, tolerance_lower)

                    if new_fibers:
                        fiber_centers.append(new_coordinate)
                        fiber_types.append("boundary")
                        fiber_count += 1
                        virtual_fiber_count += 1
                        added_fiber = True
                        fail_count = 0

                        for fiber in new_fibers:
                            fiber_centers.append(fiber)
                            fiber_types.append("periodic")
                            virtual_fiber_count += 1

                else:
                    fiber_centers.append(new_coordinate)
                    fiber_types.append("normal")
                    fiber_count += 1
                    virtual_fiber_count += 1
                    added_fiber = True
                    fail_count = 0

            trial += 1

        if not added_fiber:
            fail_count += 1

    original_fibers = [fiber for i, fiber in enumerate(fiber_centers) if fiber_types[i] != "periodic"]

    return fiber_centers, fiber_types, fiber_count, virtual_fiber_count, original_fibers


def RFR_algorithm(fiber_centers, original_fibers, target_fiber_count, fiber_radius, rve_size_x, rve_size_y, periodic_shifts, rng):
    """Reduce fiber count to target using Random Fiber Removal (RFR).

    Args:
        fiber_centers: List of all fiber center positions (incl. periodic).
        original_fibers: List of original (non-periodic) fiber positions.
        target_fiber_count: Desired number of fibers after removal.
        fiber_radius: Radius of each fiber.
        rve_size_x: RVE dimension in x-direction.
        rve_size_y: RVE dimension in y-direction.
        periodic_shifts: List of [dx, dy] shift vectors.
        rng: random.Random instance for isolated, reproducible randomness.

    Returns:
        Tuple of (updated fiber_centers, updated original_fibers).
    """
    updated_original_fibers = original_fibers.copy()
    fiber_count = len(updated_original_fibers)

    while fiber_count > target_fiber_count:
        remove_index = rng.randint(0, len(updated_original_fibers) - 1)
        fiber_to_remove = updated_original_fibers[remove_index]

        if is_fiber_at_boundary(fiber_to_remove, fiber_radius, rve_size_x, rve_size_y):
            fiber_centers = remove_fiber_with_periodicity(fiber_to_remove, fiber_centers, periodic_shifts)
            updated_original_fibers = [f for f in updated_original_fibers if not np.allclose(f, fiber_to_remove, atol = 1e-6)]
        else:
            fiber_centers.remove(fiber_to_remove)
            updated_original_fibers.remove(fiber_to_remove)

        fiber_count -= 1

    return fiber_centers, updated_original_fibers


# ================================================================
# Main function
# ================================================================
def main():
    """Run fiber distribution generation from config files."""
    from config_loader import config
    from plot_fiber_distribution import plot_comparison
    
    # Standalone mode: job_id = 0 so we match the first grid job's mesh.
    rng = derive_job_rng(config.get("algorithm.seed"), job_id = 0)

    print("=" * 60)
    print("FIBER DISTRIBUTION ALGORITHM")
    print("=" * 60, "\n")

    # Get parameters from config
    rve_size_x = config.get("rve.rve.size_x")
    rve_size_y = config.get("rve.rve.size_y")
    fiber_radius = config.get("rve.fiber.radius")
    periodic_boundary_buffer = config.get("rve.rve.periodic_boundary_buffer")

    max_fibers_count = config.get("algorithm.max_fibers_count")
    max_failed_attempts = config.get("algorithm.max_failed_attempts")
    max_trials_per_fiber = config.get("algorithm.max_trials_per_fiber")
    distance_factors = config.get("algorithm.distance_factors")
    periodic_shifts = config.get("algorithm.periodic_shifts")

    exclusion_zones = config.get_exclusion_zones(fiber_radius, rve_size_x, rve_size_y)

    csv_dir = config.get("paths.data.csv")
    enable_plotting = True
    plot_frequency = 1

    rse_args = dict(rve_size_x = rve_size_x, 
                    rve_size_y = rve_size_y,
                    fiber_radius = fiber_radius,
                    exclusion_zones = exclusion_zones,
                    max_fibers_count = max_fibers_count,
                    max_failed_attempts = max_failed_attempts,
                    max_trials_per_fiber = max_trials_per_fiber,
                    distance_factors = distance_factors,
                    periodic_boundary_buffer = periodic_boundary_buffer,
                    periodic_shifts = periodic_shifts,
                    rng = rng)

    rfr_args = dict(fiber_radius = fiber_radius,
                    rve_size_x = rve_size_x,
                    rve_size_y = rve_size_y,
                    periodic_shifts = periodic_shifts,
                    rng = rng)

    total_variants_created = 0
    volume_fraction_combinations = config.get("targets.volume_fraction_combinations")

    for vf_config in volume_fraction_combinations:
        vf_percent = vf_config["vf_percent"]
        vf_decimal = vf_percent / 100.0

        print(f"Volume Fraction: {vf_percent}%")
        print("-" * 60)

        # Calculate target fiber count
        target_fiber_count = calculate_target_fiber_count(vf_decimal, fiber_radius, rve_size_x, rve_size_y)

        RSE_variants = vf_config["RSE_variants"]
        RFR_variants = vf_config["RFR_variants"]
        print(f"Classical Mode: {RSE_variants} RSE × {RFR_variants} RFR = {RSE_variants * RFR_variants} Variant(s)\n")

        total_index = 0

        for RSE_index in tqdm(range(1, RSE_variants + 1), desc = f"RSE for VF = {vf_percent}%"):
            fiber_centers, fiber_types, _, _, original_fibers = RSE_algorithm(**rse_args)

            for RFR_index in range(1, RFR_variants + 1):
                total_index += 1

                reduced_fibers, updated_originals = RFR_algorithm(fiber_centers.copy(),
                                                                  original_fibers.copy(),
                                                                  target_fiber_count,
                                                                  **rfr_args)

                file_prefix = config.get("paths.file_naming.fiber_distribution")
                identifier = f"{file_prefix}_vf{vf_percent}_{total_index}"
                save_to_csv(reduced_fibers, fiber_radius, csv_dir, identifier)

                if enable_plotting and total_index % plot_frequency == 0:
                    plot_comparison(fiber_centers,
                                    fiber_types,
                                    reduced_fibers,
                                    fiber_radius,
                                    rve_size_x,
                                    rve_size_y,
                                    identifier,
                                    vf_percent)

        total_variants_created += total_index

    print("\n" + "-" * 60)
    print(f"SUMMARY: {total_variants_created} RVE(s) CREATED")
    print("-" * 60)


if __name__ == "__main__":
    main()
