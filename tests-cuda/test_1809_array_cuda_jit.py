# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

try:
    import numba as nb  # noqa: E402
    import numba.cuda as nb_cuda  # noqa: E402
    from numba import config

    ak.numba.register_and_check()

except ImportError:
    nb = nb_cuda = None

numbatest = pytest.mark.skipif(
    nb is None or nb_cuda is None, reason="requires the numba and numba.cuda packages"
)

threads_per_block = 128
blocks_per_grid = 12

config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def multiply(array, n, out):
    tid = nb_cuda.grid(1)
    out[tid] = array[tid] * n


@nb_cuda.jit(extensions=[ak.numba.array_view_arg_handler])
def pass_through(array, out):
    nb_cuda.grid(1)
    index = 0
    for x in range(len(array)):
        ilen = len(array[x])
        for i in range(ilen):
            out[index] = array[x][i]
            index = index + 1


@numbatest
def test_array_multiply():

    # create an ak.Array with cuda backend:
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")

    # allocate the result:
    results = nb_cuda.to_device(np.empty(4, dtype=np.int32))

    multiply[threads_per_block, blocks_per_grid](akarray, 3, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 3, 6, 9]


@numbatest
def test_ListOffsetArray():

    array = ak.Array([[0, 1], [2], [3, 4, 5]], backend="cuda")

    results = nb_cuda.to_device(np.empty(6, dtype=np.int32))

    pass_through[1, 1](array, results)

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 1, 2, 3, 4, 5]


@numbatest
def test_array_on_cpu_multiply():

    # create an ak.Array with a cpu backend:
    array = ak.Array([0, 1, 2, 3])

    # allocate the result:
    results = nb_cuda.to_device(np.empty(4, dtype=np.int32))

    with pytest.raises(TypeError):
        multiply[threads_per_block, blocks_per_grid](array, 3, results)

    multiply[threads_per_block, blocks_per_grid](
        ak.to_backend(array, backend="cuda"), 3, results
    )

    nb_cuda.synchronize()
    host_results = results.copy_to_host()

    assert ak.Array(host_results).tolist() == [0, 3, 6, 9]
