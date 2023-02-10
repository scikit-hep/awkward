# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
from awkward._connect.numba.arrayview_cuda import array_view_arg_handler

numba = pytest.importorskip("numba")
from numba import config, cuda

config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False

ak_numba_cuda = pytest.importorskip("awkward.numba_cuda")
ak_numba_cuda_arrayview = pytest.importorskip("awkward._connect.numba.arrayview_cuda")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba_cuda.register_and_check()

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# access statistics of these memory pools.
def print_mempool_stats(idx):
    print("print_mempool_stats: -------------------------")
    print("print_mempool_stats:", idx, ": mempool.used_bytes", mempool.used_bytes())
    print("print_mempool_stats:", idx, ": mempool.total_bytes", mempool.total_bytes())
    print(
        "print_mempool_stats:",
        idx,
        ": pinned_mempool.n_free_blocks",
        pinned_mempool.n_free_blocks(),
    )


# FIXME: configure the blocks
# threadsperblock = 32
# blockspergrid = 128

# threads = 64
# blocks = 64
# nthreads = blocks * threads


@cuda.jit(debug=True, opt=False, extensions=[array_view_arg_handler])
def multiply(array, n, out):
    tid = cuda.grid(1)
    print("     kernel multiply for tid...", tid, len(array))
    out[tid] = array[tid]  # * n


def test_array_multiply():

    if numba.cuda.is_available():
        print("test_array_multiply line 150: CUDA GPU is available!")
    else:
        print("test_array_multiply line 152: NO CUDA GPU...")

    numba.cuda.detect()

    print("test_array_multiply line 156: create an ak.Array with cuda backend:")
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")
    print("test_array_multiply line 158: access statistics of the memory pools:")
    print_mempool_stats(1)

    print("test_array_multiply line 161: allocate the result:")
    nthreads = 4
    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))
    print("test_array_multiply line 164: access statistics of the memory pools:")
    print_mempool_stats(2)

    print("test_array_multiply line 167: CALL kernel multiply...")
    multiply[1, 4](akarray, 3, results)
    print("test_array_multiply line 169: ...done.")
    cuda.synchronize()
    host_results = results.copy_to_host()

    print("test_array_multiply line 173: multiplied Awkward Array", host_results)
