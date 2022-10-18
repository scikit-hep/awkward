# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba import cuda  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402

ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")

ak.numba.register_and_check()


def test_arrays():
    @numba.njit
    def something(array):
        for index in range(len(array)):
            print(array[index])  # noqa: T201

    akarray = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    something(akarray)

    @numba.cuda.jit
    def something_else(array):
        index = numba.cuda.threadIdx.x
        if index > len(array):
            return

    another_akarray = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]], backend="cuda"
    )
    something_else(another_akarray)
