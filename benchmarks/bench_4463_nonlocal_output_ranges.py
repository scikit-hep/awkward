# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Compare this script on main and the migration branch in one GPU environment.

Run with the same Python/CUDA dependencies, GPU, iteration and repeat counts:
    python benchmarks/bench_4463_nonlocal_output_ranges.py --iterations 100 --repeats 5

Copy this script into the main checkout for the baseline run. Validation and
compilation warmup occur before timing. Results include median host wall time
and CUDA event time per call, including each implementation's allocations.
No speedup should be claimed until both sets of results are available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import cupy as cp
import numpy as np

import awkward as ak
from awkward._backends.cupy import CupyBackend


def measure(groups, width, iterations, repeats):
    key = ("awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",) + (
        np.int64,
    ) * 3
    operation = CupyBackend.instance()[key]
    # Prefix occupancy also makes the benchmark valid for the old raw kernel,
    # whose counting algorithm gives incorrect stops when groups have holes.
    counts = np.arange(groups) % (width + 1)
    distincts = np.where(np.arange(width)[None, :] < counts[:, None], 0, -1).astype(
        np.int64
    )
    source = cp.asarray(distincts.reshape(-1))
    starts, stops = cp.empty(groups, dtype=cp.int64), cp.empty(groups, dtype=cp.int64)

    def run():
        operation(starts, stops, source, source.size, groups)

    for _ in range(3):
        run()
    ak._connect.cuda.synchronize_cuda()
    expected_starts = np.arange(groups) * width
    np.testing.assert_array_equal(cp.asnumpy(starts), expected_starts)
    np.testing.assert_array_equal(cp.asnumpy(stops), expected_starts + counts)
    wall, dispatch, device = [], [], []
    for _ in range(repeats):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        ak._connect.cuda.synchronize_cuda()
        before = time.perf_counter()
        start.record()
        for _ in range(iterations):
            run()
        stop.record()
        dispatch.append((time.perf_counter() - before) * 1e6 / iterations)
        stop.synchronize()
        wall.append((time.perf_counter() - before) * 1e6 / iterations)
        device.append(cp.cuda.get_elapsed_time(start, stop) * 1e3 / iterations)
        ak._connect.cuda.synchronize_cuda()
    return {
        "groups": groups,
        "width": width,
        "wall_us": statistics.median(wall),
        "host_dispatch_us": statistics.median(dispatch),
        "device_us": statistics.median(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    print(
        json.dumps(
            {
                "label": args.label,
                "awkward": ak.__version__,
                "cupy": cp.__version__,
                "numpy": np.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "device": str(
                    cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
                ),
                "iterations": args.iterations,
                "repeats": args.repeats,
            }
        )
    )
    for groups, width in [
        (1, 1),
        (8, 4),
        (128, 32),
        (1024, 256),
        (4096, 256),
        (8, 65536),
    ]:
        print(json.dumps(measure(groups, width, args.iterations, args.repeats)))


if __name__ == "__main__":
    main()
