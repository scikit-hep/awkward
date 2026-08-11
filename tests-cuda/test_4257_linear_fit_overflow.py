# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# ak.linear_fit promotes integer x, y to float64 before forming x**2 and x*y, so
# integer squares/products no longer overflow at the input dtype. The promotion
# (values_astype) and the sums run on the array's own backend, so on cuda the
# whole fit stays on device -- no host transfer, no cuda-specific code path.
# Cross-checked against the CPU backend.


def _gpu(array):
    """Assert an array is still on the cuda backend, then return it."""
    assert ak.backend(array) == "cuda"
    return array


def test_linear_fit_int32_overflow_gpu():
    # 100000**2 overflows int32; y = 2x + 1. Correct only if promoted to float64.
    x = np.array([100000, 200000, 300000, 400000], dtype=np.int32)
    y = (2 * x + 1).astype(np.int32)
    gpu = ak.linear_fit(
        ak.to_backend(ak.Array(x), "cuda"), ak.to_backend(ak.Array(y), "cuda")
    )
    assert float(gpu["slope"]) == pytest.approx(2.0)
    assert float(gpu["intercept"]) == pytest.approx(1.0)


def test_linear_fit_gpu_matches_cpu():
    cases = [
        (np.array([100000, 200000, 300000], dtype=np.int32), None),
        (np.array([3_000_000_000, 4_000_000_000, 5_000_000_000], dtype=np.int64), None),
        (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64), None),
        (np.array([100000, 200000, 300000], dtype=np.int32), np.array([1.0, 2.0, 3.0])),
    ]
    for x, w in cases:
        y = (2 * x.astype(np.float64) + 1).astype(x.dtype)
        cpu = ak.linear_fit(
            ak.Array(x), ak.Array(y), weight=None if w is None else ak.Array(w)
        )
        gpu = ak.linear_fit(
            ak.to_backend(ak.Array(x), "cuda"),
            ak.to_backend(ak.Array(y), "cuda"),
            weight=None if w is None else ak.to_backend(ak.Array(w), "cuda"),
        )
        for field in ("slope", "intercept", "intercept_error", "slope_error"):
            assert float(gpu[field]) == pytest.approx(float(cpu[field]))


def test_linear_fit_jagged_axis1_gpu_matches_cpu():
    x = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [10, 20, 30, 40]]), np.int32
    )
    y = ak.values_astype(
        ak.Array([[200001, 400001, 600001], [21, 41, 61, 81]]), np.int32
    )
    cpu = ak.linear_fit(x, y, axis=1)
    gpu = ak.linear_fit(ak.to_backend(x, "cuda"), ak.to_backend(y, "cuda"), axis=1)
    for field in ("slope", "intercept"):
        got = ak.to_list(ak.to_backend(_gpu(gpu[field]), "cpu"))
        assert got == pytest.approx(ak.to_list(cpu[field]))


def test_linear_fit_stays_on_device():
    x = ak.to_backend(
        ak.values_astype(ak.Array([[100000, 200000, 300000], [1, 2, 3, 4]]), np.int32),
        "cuda",
    )
    y = ak.to_backend(
        ak.values_astype(ak.Array([[200001, 400001, 600001], [3, 5, 7, 9]]), np.int32),
        "cuda",
    )
    fit = ak.linear_fit(x, y, axis=1)
    for field in ("slope", "intercept", "intercept_error", "slope_error"):
        assert ak.backend(fit[field]) == "cuda"
