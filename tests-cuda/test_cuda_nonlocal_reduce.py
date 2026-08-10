# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# Non-innermost (axis=0, axis=1 on deeper) reductions on the cuda backend go
# through the "non-local" reduce path. In the offsets pipeline that path invokes
# three ListOffsetArray transpose kernels -- reduce_nonlocal_maxcount_offsetscopy,
# reduce_nonlocal_preparenext, reduce_nonlocal_outstartsstops -- implemented via
# cuda.compute in _connect/cuda/_compute.py (outoffsets/nextstarts are taken
# directly from the offsets, so the old parents->offsets converters are not
# called). Here we cross-check GPU vs CPU for the non-positional reducers
# (sum/count/mean/var/std) that use them.


def _match(gpu_result, cpu_result):
    return ak.almost_equal(ak.to_backend(gpu_result, "cpu"), cpu_result)


@pytest.mark.parametrize("op", [ak.sum, ak.count, ak.mean, ak.var, ak.std])
def test_nonlocal_reduce_axis0_matches_cpu(op):
    # ragged, so axis=0 reduces column-wise across rows of differing length.
    cpu = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
    gpu = ak.to_backend(cpu, "cuda")
    assert _match(op(gpu, axis=0), op(cpu, axis=0))
    assert ak.backend(op(gpu, axis=0)) == "cuda"


@pytest.mark.parametrize("op", [ak.sum, ak.count, ak.mean, ak.var])
def test_nonlocal_reduce_three_deep_axis1_matches_cpu(op):
    # axis=1 on a 3-deep array is non-innermost -> non-local path.
    cpu = ak.Array([[[1.0, 2], [3, 4, 5]], [[6.0], [7, 8]]])
    gpu = ak.to_backend(cpu, "cuda")
    assert _match(op(gpu, axis=1), op(cpu, axis=1))


def test_nonlocal_reduce_int_axis0_overflow_safe():
    cpu = ak.values_astype(
        ak.Array([[100000, 200000], [300000, 400000, 500000]]), np.int32
    )
    gpu = ak.to_backend(cpu, "cuda")
    # var at axis=0 (non-innermost) uses the two-pass path + non-local kernels.
    assert _match(ak.var(gpu, axis=0), ak.var(cpu, axis=0))
    assert _match(ak.sum(gpu, axis=0), ak.sum(cpu, axis=0))


def test_nonlocal_reduce_with_empty_sublists():
    cpu = ak.Array([[1.0, 2.0, 3.0], [], [6.0, 7.0]])
    gpu = ak.to_backend(cpu, "cuda")
    for op in (ak.sum, ak.count, ak.mean):
        assert _match(op(gpu, axis=0), op(cpu, axis=0))
