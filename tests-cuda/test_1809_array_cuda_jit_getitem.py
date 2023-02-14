# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
from awkward._connect.numba.arrayview_cuda import array_view_arg_handler

try:
    import numba as nb  # noqa: E402
    import numba.cuda as nb_cuda  # noqa: E402
except ImportError:
    nb = nb_cuda = None

numbatest = pytest.mark.skipif(
    nb is None or nb_cuda is None, reason="requires the numba and numba.cuda packages"
)

ak_numba_cuda = pytest.importorskip("awkward.numba_cuda")
ak_numba_cuda_arrayview = pytest.importorskip("awkward._connect.numba.arrayview_cuda")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba_cuda.register_and_check()

threadsperblock = 128
blockspergrid = 1


@nb_cuda.jit(extensions=[array_view_arg_handler])
def multiply(array, n, out):
    tid = nb_cuda.grid(1)
    if tid < len(array):
        out[tid] = array[tid] * n


@numbatest
def test_array_multiply():

    if nb_cuda.is_available():
        print("CUDA GPU is available!")
    else:
        print("NO CUDA GPU...")

    nb_cuda.detect()

    # create an ak.Array with cuda backend:
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")

    # allocate the result:
    results = nb_cuda.to_device(np.empty(4, dtype=np.int32))

    multiply[threadsperblock, blockspergrid](akarray, 3, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 3, 6, 9]
