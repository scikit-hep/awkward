# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Compare ListOffsetArray kernels on main and the cuda.compute branch.

Copy this script into each checkout. Use the same GPU, dependencies, rows,
width, iteration and repeat counts. Validation and compilation precede timing;
no speedup is assumed. The deprecated parents converter is measured separately.
"""

import argparse
import statistics
import time

import cupy as cp
import numpy as np

import awkward._connect.cuda as cuda
from awkward._backends.cupy import CupyBackend
from awkward._backends.numpy import NumpyBackend


def workloads(rows, width):
    offsets = np.arange(rows + 1, dtype=np.int64) * width
    indexes = np.arange(rows * width, dtype=np.int64)
    missing = np.where(indexes % 3 == 0, -1, indexes)
    # Prefix occupancy makes the old raw nonlocal kernel a valid baseline.
    counts = np.arange(rows) % (width + 1)
    distincts = np.where(np.arange(width)[None, :] < counts[:, None], 0, -1)
    return [
        (
            "flatten_offsets",
            [
                np.empty(rows + 1, dtype=np.int64),
                np.arange(rows + 1, dtype=np.int64),
                rows + 1,
                offsets,
            ],
            [0],
        ),
        ("toRegularArray", [np.empty(1, dtype=np.int64), offsets, rows + 1], [0]),
        (
            "rpad_axis1",
            [np.empty(rows * (width + 2), dtype=np.int64), offsets, rows, width + 2],
            [0],
        ),
        (
            "drop_none_indexes",
            [
                np.empty(rows + 1, dtype=np.int64),
                missing,
                offsets,
                rows + 1,
                indexes.size,
            ],
            [0],
        ),
        (
            "reduce_nonlocal_outstartsstops_64",
            [
                np.empty(rows, dtype=np.int64),
                np.empty(rows, dtype=np.int64),
                distincts.reshape(-1).astype(np.int64),
                indexes.size,
                rows,
            ],
            [0, 1],
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if min(args.rows, args.width, args.iterations, args.repeats) <= 0:
        parser.error("counts must be positive")
    print("label,kernel,rows,width,host_us,total_us,stream_us")
    for name, values, outputs in workloads(args.rows, args.width):
        key = ("awkward_ListOffsetArray_" + name,) + (np.int64,) * sum(
            isinstance(x, np.ndarray) for x in values
        )
        gpu_values = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in values]
        error = NumpyBackend.instance()[key](*values)
        assert not error.str
        kernel = CupyBackend.instance()[key]
        for _ in range(3):
            error = kernel(*gpu_values)
            assert error is None or not error.str
        cuda.synchronize_cuda()
        for i in outputs:
            np.testing.assert_array_equal(cp.asnumpy(gpu_values[i]), values[i])
        samples = []
        for _ in range(args.repeats):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            before = time.perf_counter()
            start.record()
            for _ in range(args.iterations):
                kernel(*gpu_values)
            host = time.perf_counter() - before
            stop.record()
            stop.synchronize()
            cuda.synchronize_cuda()
            total = time.perf_counter() - before
            samples.append(
                (
                    host * 1e6 / args.iterations,
                    total * 1e6 / args.iterations,
                    cp.cuda.get_elapsed_time(start, stop) * 1e3 / args.iterations,
                )
            )
        medians = [statistics.median(x[i] for x in samples) for i in range(3)]
        print(
            f"{args.label},{name},{args.rows},{args.width},"
            + ",".join(f"{x:.3f}" for x in medians)
        )


if __name__ == "__main__":
    main()
