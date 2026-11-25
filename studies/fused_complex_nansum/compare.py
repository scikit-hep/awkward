# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np  # noqa: TID251

import awkward as ak
from awkward.operations import ak_mean, ak_sum

parser = argparse.ArgumentParser()
parser.add_argument("--worker", choices=["baseline", "fused"])
parser.add_argument("--before-library", type=Path)
parser.add_argument(
    "--patterns",
    nargs="+",
    choices=[
        "finite",
        "strided",
        "last",
        "all",
        "first-row",
        "last-row",
        "sparse-rows",
    ],
    default=["finite", "strided"],
)
parser.add_argument("--output", type=Path, default=Path("results/fused-complex-nansum"))
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
if args.worker:
    variant = args.worker
    if variant == "baseline" and args.before_library:
        import ctypes

        import awkward_cpp._kernel_signatures
        import awkward_cpp.cpu_kernels

        from awkward._backends.numpy import NumpyBackend

        library = ctypes.CDLL(str(args.before_library.resolve()))
        awkward_cpp.cpu_kernels.kernel = awkward_cpp._kernel_signatures.by_signature(
            library
        )
        NumpyBackend.instance()._kernels.clear()
    elif variant == "baseline":
        source = subprocess.check_output(
            ["git", "show", "82615e865:src/awkward/_complex_nan.py"], text=True
        )
        namespace = {"__name__": "baseline_complex_nan"}
        exec(compile(source, "baseline_complex_nan.py", "exec"), namespace)
        fn = namespace["try_reduce_dense"]
        ak_sum.try_reduce_dense = ak_mean.try_reduce_dense = fn
    rows = []
    for shape in [(4, 8), (16384, 16), (256, 1024), (16, 100000)]:
        for dtype in [np.complex64, np.complex128]:
            for pattern in args.patterns:
                data = np.full(shape, 0.1 + 0.1j, dtype=dtype)
                if pattern == "strided":
                    data[:, ::100] = complex(1, np.nan)
                elif pattern == "last":
                    data[:, -1] = complex(1, np.nan)
                elif pattern == "all":
                    data[:] = complex(np.nan, 1)
                elif pattern == "first-row":
                    data[0, -1] = complex(1, np.nan)
                elif pattern == "last-row":
                    data[-1, -1] = complex(1, np.nan)
                elif pattern == "sparse-rows":
                    data[::100, -1] = complex(1, np.nan)
                source = ak.Array(data)
                for op in [ak.nansum, ak.nanmean]:
                    ordinary = ak.sum if op is ak.nansum else ak.mean
                    expected = ordinary(
                        ak.nan_to_none(source), axis=-1, mask_identity=op is ak.nanmean
                    )
                    np.testing.assert_array_equal(
                        ak.to_numpy(op(source, axis=-1)), ak.to_numpy(expected)
                    )
                    for _ in range(3):
                        op(source, axis=-1)
                    samples = []
                    for _ in range(7):
                        t = time.perf_counter_ns()
                        for _ in range(5):
                            op(source, axis=-1)
                        samples.append((time.perf_counter_ns() - t) / 5000)
                    rows.append(
                        {
                            "case": f"{shape[0]}x{shape[1]}-{np.dtype(dtype).name}-{pattern}-{op.__name__}",
                            "us": statistics.median(samples),
                            "samples": samples,
                        }
                    )
    print(  # noqa: T201
        json.dumps(
            {
                "variant": variant,
                "rows": rows,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "source": ak.__file__,
            }
        )
    )
    sys.exit()
results = {"baseline": [], "fused": []}
for pair in range(3):
    for variant in ["baseline", "fused"] if pair % 2 == 0 else ["fused", "baseline"]:
        r = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    variant,
                    "--output",
                    str(out),
                    "--patterns",
                    *args.patterns,
                ]
                + (
                    ["--before-library", str(args.before_library)]
                    if args.before_library
                    else []
                ),
                text=True,
            )
        )
        (out / f"{pair + 1}-{variant}.json").write_text(json.dumps(r, indent=2))
        results[variant].append(r)
lines = [
    "# Fused native complex NaN prototype",
    "",
    (
        "Baseline: previous native library " + str(args.before_library)
        if args.before_library
        else "Baseline: PR #4549 helper at 82615e865"
    )
    + "; candidate: current fused native sum/count kernel. Three alternating fresh-process pairs (B/C, C/B, B/C), seven batches of five calls, three warmups. Inputs and exact parity checks outside timing. No profiling. NumPy matrix inputs, axis=-1, default options. Strided NaNs: every 100th entry per row; last: only the final entry per row; all: every entry; first-row/last-row: one NaN in that row; sparse-rows: one NaN per 100 rows. All outputs checked exactly against ordinary masked sum/mean. Local CPU measurements; not against main.",
    "",
    "| Case | Baseline µs | Fused µs | Candidate / baseline |",
    "| --- | ---: | ---: | ---: |",
]
for i, case in enumerate(results["baseline"][0]["rows"]):
    a = statistics.median(r["rows"][i]["us"] for r in results["baseline"])
    b = statistics.median(r["rows"][i]["us"] for r in results["fused"])
    lines.append(f"| {case['case']} | {a:.1f} | {b:.1f} | {b / a:.3f} |")
(out / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))  # noqa: T201
