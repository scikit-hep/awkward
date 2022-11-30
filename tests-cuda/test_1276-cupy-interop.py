# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp
import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_cupy_interop():
    c = cp.arange(10)
    n = np.arange(10)
    cupy_index_arr = ak.index.Index64(c)
    np_index_arr = ak.index.Index64(n)

    cupy = ak.nplikes.Cupy.instance()
    numpy = ak.nplikes.Numpy.instance()

    # GPU->CPU
    assert ak.to_list(np.asarray(cupy_index_arr.to_nplike(cupy))) == ak.to_list(
        np.asarray(np_index_arr)
    )
    # CPU->CPU
    assert ak.to_list(np.asarray(np_index_arr.to_nplike(cupy))) == ak.to_list(
        np.asarray(np_index_arr)
    )
    # CPU->GPU->CPU
    assert ak.to_list(np.asarray(np_index_arr)) == ak.to_list(
        np.asarray(np_index_arr.to_nplike(cupy).to_backend(numpy))
    )
