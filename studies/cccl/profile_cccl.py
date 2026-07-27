"""
To run:

nsys profile -t cuda,nvtx python profile_cccl.py
"""

import awkward as ak
import numpy as np
import cProfile
import sys
from pathlib import Path
import cupy as cp
import nvtx

# Add current directory to path to import playground
sys.path.insert(0, str(Path(__file__).parent))
from playground import physics_analysis_cccl  # noqa: E402


def generate_random_events(num_events=1000000, seed=42):
    """
    Generate random physics events with electrons and muons.

    Args:
        num_events: Number of events to generate
        seed: Random seed for reproducibility

    Returns:
        Awkward Array with structure matching playground.py events
    """
    np.random.seed(seed)

    # Generate random counts for electrons and muons per event (0-10 each)
    num_electrons_per_event = np.random.randint(1, 21, size=num_events)
    num_muons_per_event = np.random.randint(1, 21, size=num_events)

    total_electrons = np.sum(num_electrons_per_event)
    total_muons = np.sum(num_muons_per_event)

    # Generate random physics values for all electrons
    electron_pts = np.random.uniform(10, 100, size=total_electrons)
    electron_etas = np.random.uniform(-3, 3, size=total_electrons)
    electron_phis = np.random.uniform(0, 2*np.pi, size=total_electrons)

    # Generate random physics values for all muons
    muon_pts = np.random.uniform(10, 100, size=total_muons)
    muon_etas = np.random.uniform(-3, 3, size=total_muons)
    muon_phis = np.random.uniform(0, 2*np.pi, size=total_muons)

    # Build awkward arrays with jagged structure
    electrons = ak.Array({
        "pt": ak.unflatten(electron_pts, num_electrons_per_event),
        "eta": ak.unflatten(electron_etas, num_electrons_per_event),
        "phi": ak.unflatten(electron_phis, num_electrons_per_event),
    })

    muons = ak.Array({
        "pt": ak.unflatten(muon_pts, num_muons_per_event),
        "eta": ak.unflatten(muon_etas, num_muons_per_event),
        "phi": ak.unflatten(muon_phis, num_muons_per_event),
    })

    events = ak.zip({"electrons": electrons, "muons": muons}, depth_limit=1)

    print(f"Generated {num_events:,} events")
    print(f"  Total electrons: {total_electrons:,}")
    print(f"  Total muons: {total_muons:,}")
    print()

    return events


if __name__ == "__main__":
    # Generate events
    print("Generating events...")
    events = generate_random_events(num_events=2**24)
    events_gpu = ak.to_backend(events, "cuda")
    cp.cuda.Device().synchronize()

    # Warmup run (not profiled)
    print("Warming up physics_analysis_cccl...")
    _ = physics_analysis_cccl(events_gpu)
    print("Warmup complete.\n")

    # Profile the actual run
    cp.cuda.Device().synchronize()

    print("Profiling physics_analysis_cccl (non-warmup run)...")

    # Use runctx to avoid conflicts with existing profilers
    profile_file = "cccl_profile.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    with nvtx.annotate("physics_analysis_cccl"):
        physics_analysis_cccl(events_gpu)
    profiler.disable()
    profiler.dump_stats(profile_file)

    print(f"\nProfile saved to: {profile_file}")
    cp.cuda.Device().synchronize()
