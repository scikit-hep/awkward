# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

nb = pytest.importorskip("numba")
nb_cuda = pytest.importorskip("numba.cuda")

from numba import types  # noqa: E402

nb.config.CUDA_LOW_OCCUPANCY_WARNINGS = False
nb.config.CUDA_WARN_ON_IMPLICIT_COPY = False


try:
    ak.numba.register_and_check()
except ImportError:
    pytest.skip(reason="too old Numba version", allow_module_level=True)


def test_array_typed():
    # create an ak.Array with a cuda backend:
    gpu_arr_type = ak.Array([[[0, 1], [2, 3]], [[4, 5]]], backend="cuda").numba_type

    @nb.cuda.jit(types.void(gpu_arr_type), extensions=[ak.numba.cuda])
    def cuda_kernel(arr):
        return None

    array = ak.Array([[[0, 1], [2, 3]], [[4, 5]]], backend="cuda")
    cuda_kernel[1024, 1024](array)

    other_array = ak.Array(
        [[1.1, 2.2, 3.3], [None, 4.4], None, [None, 5.5]], backend="cuda"
    )

    with pytest.raises(nb.core.errors.NumbaTypeError):
        cuda_kernel[1024, 1024](other_array)
