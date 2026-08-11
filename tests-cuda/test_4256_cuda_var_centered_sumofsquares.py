# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# ak.var/ak.std on the cuda backend, unweighted, at the innermost axis use the
# fused centered sum-of-squares reducer (awkward_reduce_centered_sumofsquares in
# _connect/cuda/_compute.py): the per-segment mean is gathered to each element
# and a ZipIterator + TransformIterator forms (x - mean)**2 in float64 on the fly,
# feeding a PLUS segmented_reduce -- no deviation buffer and no mean back-broadcast.
# Cross-checked against the CPU backend. This file covers only the innermost
# (fused) path; the non-innermost axis (not fused, two-pass) is covered by
# tests-cuda/test_cuda_nonlocal_reduce.py.

DTYPES = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "int64",
    "float32",
    "float64",
]


def _gpu(array):
    """Assert an array is still on the cuda backend, then return it."""
    assert ak.backend(array) == "cuda"
    return array


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_std_innermost_matches_cpu(dtype):
    segs = [[3, 1, 2, 4], [], [40, 50], [7]]
    cpu = ak.values_astype(ak.Array(segs), dtype)
    gpu = ak.to_backend(cpu, "cuda")

    for op in (ak.var, ak.std):
        out_cpu = ak.to_list(op(cpu, axis=-1))
        out_gpu = ak.to_list(ak.to_backend(_gpu(op(gpu, axis=-1)), "cpu"))
        for a, b in zip(out_cpu, out_gpu, strict=True):
            if a is None or (isinstance(a, float) and np.isnan(a)):
                assert b is None or np.isnan(b)
            else:
                assert b == pytest.approx(a)


def test_var_int32_overflow_innermost():
    # (value - mean)**2 overflows int32; the kernel centres in float64.
    cpu = ak.values_astype(ak.Array([[100000, 200000, 300000]]), np.int32)
    gpu = ak.to_backend(cpu, "cuda")
    result = ak.to_list(ak.to_backend(_gpu(ak.var(gpu, axis=-1)), "cpu"))[0]
    assert result == pytest.approx(np.var([100000.0, 200000.0, 300000.0]))


def test_var_float32_cancellation_innermost():
    cpu = ak.values_astype(ak.Array([[1e7, 1e7 + 1, 1e7 + 2]]), np.float32)
    gpu = ak.to_backend(cpu, "cuda")
    result = ak.to_list(ak.to_backend(_gpu(ak.var(gpu, axis=-1)), "cpu"))[0]
    assert result == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_var_empty_sublist_masked_innermost():
    cpu = ak.Array([[1.0, 2, 3], [], [6.0, 7]])
    gpu = ak.to_backend(cpu, "cuda")
    out = ak.to_list(
        ak.to_backend(_gpu(ak.var(gpu, axis=-1, mask_identity=True)), "cpu")
    )
    assert out[0] == pytest.approx(np.var([1, 2, 3]))
    assert out[1] is None
    assert out[2] == pytest.approx(np.var([6, 7]))


def test_var_three_deep_innermost_matches_cpu():
    segs = [[[1.0, 2], [3, 4, 5]], [[6.0, 7]]]
    cpu = ak.Array(segs)
    gpu = ak.to_backend(cpu, "cuda")
    out_cpu = ak.var(cpu, axis=-1)
    out_gpu = ak.to_backend(_gpu(ak.var(gpu, axis=-1)), "cpu")
    # nested result: compare with structure+tolerance (pytest.approx is flat-only)
    assert ak.almost_equal(out_gpu, out_cpu)


# NB: a non-innermost axis (e.g. axis=0) is not fused; it uses the general
# two-pass reduce. That path's non-local reduce kernel
# (awkward_ListOffsetArray_reduce_nonlocal_preparenext_64) is implemented on the
# CUDA backend by this PR and exercised in test_cuda_nonlocal_reduce.py. This
# file focuses on the innermost (fused) centered-sum-of-squares path.


def test_var_std_stay_on_device():
    x = ak.to_backend(ak.values_astype(ak.Array([[3, 1, 2], [4, 5]]), np.int32), "cuda")
    assert ak.backend(ak.var(x, axis=-1)) == "cuda"
    assert ak.backend(ak.std(x, axis=-1)) == "cuda"
