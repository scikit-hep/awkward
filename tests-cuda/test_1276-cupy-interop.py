# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp
import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_cupy_interop():
    cupy = ak._nplikes.Cupy.instance()
    numpy = ak._nplikes.Numpy.instance()

    cupy_index_arr = ak.index.Index64(cp.arange(10))
    assert cupy_index_arr.nplike is cupy

    np_index_arr = ak.index.Index64(np.arange(10))
    assert np_index_arr.nplike is numpy

    cupy_as_numpy = cupy_index_arr.to_nplike(numpy)
    assert isinstance(cupy_as_numpy.data, np.ndarray) and not isinstance(
        cupy_as_numpy.data, cp.ndarray
    )

    numpy_as_cupy = np_index_arr.to_nplike(cupy)
    assert isinstance(numpy_as_cupy.data, cp.ndarray)

    # GPU->CPU
    assert cupy_as_numpy.data.tolist() == np.asarray(np_index_arr).tolist()
    # CPU->CPU
    assert numpy_as_cupy.data.tolist() == np.asarray(np_index_arr).tolist()

    numpy_round_trip = numpy_as_cupy.to_nplike(numpy)
    assert isinstance(numpy_round_trip.data, np.ndarray)

    # CPU->GPU->CPU
    assert numpy_round_trip.data.tolist() == np.asarray(np_index_arr).tolist()
