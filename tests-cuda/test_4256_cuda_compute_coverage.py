# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# Coverage for the cuda.compute helpers in _connect/cuda/_compute.py:
#   * awkward_ListOffsetArray_reduce_nonlocal_preparenext_64 -- the empty-input
#     early return (nextlen == 0 or nbins == 0) and the main path.
#   * awkward_reduce_centered_sumofsquares._centered_square -- the (x-mean)**2
#     device op.
# The preparenext kernel runs for a NON-innermost reduce (axis=0); these tests
# also exercise the axis=0 reduce end-to-end (its outstartsstops step is served
# by the compiled .cu template kernel). The centered op runs for an innermost
# ak.var/ak.std. All results are cross-checked vs the CPU backend.


def _match(gpu_out, cpu_out):
    return ak.almost_equal(ak.to_backend(gpu_out, "cpu"), cpu_out)


@pytest.mark.parametrize("op", [ak.sum, ak.count, ak.mean])
def test_nonlocal_axis0_ragged_main(op):
    # Ragged axis=0 -> preparenext main path, then the axis=0 reduce end-to-end.
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
    cpu = ak.Array(rows)
    gpu = ak.to_backend(cpu, "cuda")
    assert _match(op(gpu, axis=0), op(cpu, axis=0))


def test_nonlocal_axis0_all_empty():
    # All-empty sublists axis=0 -> maxcount == 0, so preparenext hits the
    # `nextlen == 0 or nbins == 0` early return. Result is empty.
    cpu = ak.Array([[], [], []])
    gpu = ak.to_backend(cpu, "cuda")
    out = ak.sum(gpu, axis=0)
    assert ak.backend(out) == "cuda"
    assert ak.to_list(ak.to_backend(out, "cpu")) == []
    assert ak.to_list(ak.to_backend(ak.count(gpu, axis=0), "cpu")) == []


def test_nonlocal_axis0_some_empty_rows():
    # Empty rows interleaved with non-empty -> preparenext with a zero-length
    # segment, then the axis=0 reduce with a bin whose present-count differs.
    rows = [[1.0, 2.0], [], [3.0, 4.0, 5.0]]
    cpu = ak.Array(rows)
    gpu = ak.to_backend(cpu, "cuda")
    for op in (ak.sum, ak.count):
        assert _match(op(gpu, axis=0), op(cpu, axis=0))


def test_nonlocal_axis0_single_row():
    # outlength == 1: exercises the preparenext single-outer-bin path and the
    # minimal axis=0 reduce.
    cpu = ak.Array([[1.0, 2.0, 3.0]])
    gpu = ak.to_backend(cpu, "cuda")
    assert _match(ak.sum(gpu, axis=0), ak.sum(cpu, axis=0))


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64"])
def test_centered_square_via_var_innermost(dtype):
    # Innermost ak.var routes through awkward_reduce_centered_sumofsquares, whose
    # _centered_square device op computes (x - mean)**2 (lines 780-781). Integer
    # input exercises the float64-promoting subtraction there.
    cpu = ak.values_astype(ak.Array([[1, 2, 3, 4], [5, 6], [7]]), dtype)
    gpu = ak.to_backend(cpu, "cuda")
    for op in (ak.var, ak.std):
        assert _match(op(gpu, axis=-1), op(cpu, axis=-1))


def test_centered_square_overflow_safe():
    # (value - mean) is formed in float64 in _centered_square, so int32 values
    # whose squared deviations overflow int32 still give the right variance.
    cpu = ak.values_astype(ak.Array([[100000, 200000, 300000]]), np.int32)
    gpu = ak.to_backend(cpu, "cuda")
    got = ak.to_list(ak.to_backend(ak.var(gpu, axis=-1), "cpu"))[0]
    assert got == pytest.approx(np.var([100000.0, 200000.0, 300000.0]))
