import sys
import awkward as ak
import numpy as np
import cupy as cp
import time
from pathlib import Path

# Add current directory to path to import playground
sys.path.insert(0, str(Path(__file__).parent))
from playground import physics_analysis, physics_analysis_gpu, physics_analysis_cccl  # noqa: E402


def generate_random_events(num_events=50000, seed=42):
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
    num_electrons_per_event = np.random.randint(0, 11, size=num_events)
    num_muons_per_event = np.random.randint(0, 11, size=num_events)

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
    print(f"  Avg electrons per event: {total_electrons/num_events:.2f}")
    print(f"  Avg muons per event: {total_muons/num_events:.2f}")
    print()

    return events


def benchmark_analysis(events):
    """
    Benchmark the three analysis approaches with warmup runs.
    Warmup runs are excluded from timing (only measure steady-state performance).

    Args:
        events: Awkward Array of events to analyze
    """
    print("=" * 60)
    print("BENCHMARKING PHYSICS ANALYSIS")
    print("=" * 60)
    print()

    # Warmup and benchmark CPU version
    print("Warming up physics_analysis (CPU)...")
    _ = physics_analysis(events)
    print("Running physics_analysis (CPU)...")
    start = time.perf_counter()
    result_cpu = physics_analysis(events)
    time_cpu = time.perf_counter() - start
    print(f"  Time: {time_cpu:.4f} seconds")
    print()

    events_gpu = ak.to_backend(events, "cuda")

    # Warmup and benchmark GPU native version
    print("Warming up physics_analysis_gpu (GPU native)...")
    _ = physics_analysis_gpu(events_gpu)
    print("Running physics_analysis_gpu (GPU native)...")
    start = time.perf_counter()
    result_gpu = physics_analysis_gpu(events_gpu)
    cp.cuda.Device().synchronize()
    time_gpu = time.perf_counter() - start
    print(f"  Time: {time_gpu:.4f} seconds")
    print()

    # Warmup and benchmark CCCL version
    print("Warming up physics_analysis_cccl (CCCL)...")
    _ = physics_analysis_cccl(events_gpu)
    print("Running physics_analysis_cccl (CCCL)...")
    start = time.perf_counter()
    result_cccl = physics_analysis_cccl(events_gpu)
    cp.cuda.Device().synchronize()
    time_cccl = time.perf_counter() - start
    print(f"  Time: {time_cccl:.4f} seconds")
    print()

    # Display summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"CPU:          {time_cpu:.4f} seconds (baseline)")
    print(
        f"GPU native:   {time_gpu:.4f} seconds ({time_cpu/time_gpu:.2f}x speedup)")
    print(
        f"CCCL:         {time_cccl:.4f} seconds ({time_cpu/time_cccl:.2f}x speedup)")
    print()

    # Print sample results to verify correctness
    print("Sample results (first 5 events with 2 electrons):")
    print(f"  CPU electrons:    {result_cpu['electron'][:5]}")
    print(f"  GPU electrons:    {result_gpu['electron'][:5]}")
    print(f"  CCCL electrons:   {result_cccl['electron'][:5]}")
    print()

    # Check correctness
    print("Checking correctness...")
    cp.testing.assert_allclose(
        result_cpu['electron'], result_cccl['electron'])
    print("Correctness check passed")
    print()


if __name__ == "__main__":
    # Generate random events at scale
    events = generate_random_events(num_events=2**24)

    # Run benchmarks
    benchmark_analysis(events)
