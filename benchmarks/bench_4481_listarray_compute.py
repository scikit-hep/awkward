# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Compare warmed ListArray workloads on main and the CCCL migration branch.

Copy this script to both checkouts and run on the same GPU/environment with
matching options. Report results for many short lists and a few long lists.
No speedup is assumed: correctness and comparative GPU timings gate merging.
"""

import argparse
import statistics
import time

import cupy as cp
import numpy as np

import awkward as ak
import awkward._connect.cuda as cuda


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
    starts = np.arange(args.rows, dtype=np.int64) * args.width
    source = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64(starts),
            ak.index.Index64(starts + args.width),
            ak.contents.NumpyArray(np.arange(args.rows * args.width)),
        )
    )
    jagged = ak.Array([[0, args.width - 1] if i % 2 else [0] for i in range(args.rows)])
    missing = ak.Array(
        [[0, None, args.width - 1] if i % 2 else [None, 0] for i in range(args.rows)]
    )
    array = ak.to_backend(source, "cuda")
    gpu_jagged = ak.to_backend(jagged, "cuda")
    gpu_missing = ak.to_backend(missing, "cuda")
    operations = {
        "scalar": (lambda x, j, m: x[:, 0]),
        "array": (lambda x, j, m: x[:, [0, -1]]),
        "range": (lambda x, j, m: x[:, ::2]),
        "reverse": (lambda x, j, m: x[:, ::-1]),
        "jagged": (lambda x, j, m: x[j]),
        "jagged_missing": (lambda x, j, m: x[m]),
        "pad": (lambda x, j, m: ak.pad_none(x, args.width + 2)),
    }
    print("label,operation,rows,width,host_us,total_us,stream_us")
    for name, operation in operations.items():
        expected = operation(source, jagged, missing).to_list()
        result = operation(array, gpu_jagged, gpu_missing)
        cuda.synchronize_cuda()
        assert result.to_list() == expected
        for _ in range(3):
            operation(array, gpu_jagged, gpu_missing)
        cuda.synchronize_cuda()
        samples = []
        for _ in range(args.repeats):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            before = time.perf_counter()
            start.record()
            for _ in range(args.iterations):
                result = operation(array, gpu_jagged, gpu_missing)
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
        values = [statistics.median(x[i] for x in samples) for i in range(3)]
        print(
            f"{args.label},{name},{args.rows},{args.width},"
            + ",".join(f"{x:.3f}" for x in values)
        )


if __name__ == "__main__":
    main()
