# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba import config, cuda, types  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402

config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False

ak_numba = pytest.importorskip("awkward.numba")
ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba.register_and_check()


def test_array_multiply_numba():
    @numba.njit(debug=True)
    def mul(array, out, n):
        print("START mul jitted function--->>>")
        for i in range(len(array)):
            out[i] = array[i] * n
        print("<--- END.")

    akarray = ak.Array([0, 1, 2, 3])  ###, backend="cuda")
    out = np.zeros(4, dtype=np.int64)

    mul(akarray, out, 3)
    print("NUMBA jit multiply", out)
