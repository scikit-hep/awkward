# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# float16 has no compiled reducer/sort kernel. The fix lives in the
# backend-agnostic NumpyArray reduce/sort methods (cast float16 -> float32,
# operate, cast value-preserving results back), so it works on the cuda backend
# too: the cast runs on-device via the cupy nplike, so nothing round-trips to the
# host. (float128/complex256 are CPU-only NumPy dtypes and never reach cupy.)
# Cross-checked against the CPU backend.

ROWS = [[1.0, 2.0, 3.0], [4.0, 5.0]]  # exact in float16


def _f16(backend):
    return ak.to_backend(ak.values_astype(ak.Array(ROWS), np.float16), backend)


@pytest.mark.parametrize(
    "op",
    [
        ak.sum,
        ak.prod,
        ak.min,
        ak.max,
        ak.any,
        ak.all,
        ak.argmin,
        ak.argmax,
        ak.count_nonzero,
    ],
)
def test_float16_reducers_gpu_matches_cpu(op):
    cpu = _f16("cpu")
    gpu = _f16("cuda")
    out_gpu = op(gpu, axis=1)
    assert ak.backend(out_gpu) == "cuda"  # stayed on device
    assert ak.almost_equal(ak.to_backend(out_gpu, "cpu"), op(cpu, axis=1))


def test_float16_sort_argsort_gpu_matches_cpu():
    cpu = _f16("cpu")
    gpu = _f16("cuda")
    for op in (ak.sort, ak.argsort):
        out_gpu = op(gpu, axis=1)
        assert ak.backend(out_gpu) == "cuda"
        assert ak.almost_equal(ak.to_backend(out_gpu, "cpu"), op(cpu, axis=1))
    # sort preserves float16
    assert "float16" in str(ak.sort(gpu, axis=1).type)


def test_float16_reducers_stay_on_device():
    gpu = _f16("cuda")
    for op in (ak.sum, ak.prod, ak.min, ak.max, ak.sort, ak.argsort):
        assert ak.backend(op(gpu, axis=1)) == "cuda"


def test_complex_sort_raises_typeerror_on_gpu():
    gpu = ak.to_backend(ak.Array([[1 + 1j, 2 + 0j], [3 - 1j]]), "cuda")
    with pytest.raises(TypeError, match="not supported"):
        ak.sort(gpu, axis=1)
    with pytest.raises(TypeError, match="not supported"):
        ak.argsort(gpu, axis=1)
