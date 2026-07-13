"""Seed derivation for reproducible, thread-safe job RNGs.

Single source of truth for the mapping (super_seed, job_id) -> per-job RNG.
All callers MUST use these helpers instead of calling `random.seed()` or
`np.random.seed()` directly — global seeds race between threads and are
therefore not reproducible in parallelized QUEENS workflows.
"""

import random

import numpy as np

# ================================================================
# Seed Generator
# ================================================================
def derive_job_rng(super_seed, job_id) -> random.Random:
    """Build an isolated RNG for a single job.

    Uses numpy.random.SeedSequence to derive a child seed
    from the master super_seed. Each (super_seed, job_id) pair maps
    deterministically to the same RNG state — on any machine, in any
    thread, regardless of execution order.

    Args:
        super_seed: Master seed (from config `algorithm.seed`).
        job_id:     QUEENS job ID.

    Returns:
        random.Random instance with isolated state.
    """
    ss = np.random.SeedSequence(entropy = int(super_seed), spawn_key = (int(job_id),))
    seed_int = int(ss.generate_state(1, dtype = np.uint32)[0])
    return random.Random(seed_int)