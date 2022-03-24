# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import cupy as cp  # noqa: F401
import awkward as ak  # noqa: F401


def test_cupy_interop():
    c = cp.arange(10)
    n = np.arange(10)
    cupy_index_arr = ak._v2.index.Index64(c)
    np_index_arr = ak._v2.index.Index64(n)

    # GPU->CPU
    assert ak._v2.to_list(
        np.asarray(cupy_index_arr.to_backend("cpu"))
    ) == ak._v2.to_list(np.asarray(np_index_arr))
    # CPU->CPU
    assert ak._v2.to_list(np.asarray(np_index_arr.to_backend("cpu"))) == ak._v2.to_list(
        np.asarray(np_index_arr)
    )
    # CPU->GPU->CPU
    assert ak._v2.to_list(np.asarray(np_index_arr)) == ak._v2.to_list(
        np.asarray(np_index_arr.to_backend("cuda").to_backend("cpu"))
    )
