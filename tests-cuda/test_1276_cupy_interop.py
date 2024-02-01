# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

try:
    ak.numba.register_and_check()
except ImportError:
    pytest.skip(reason="too old Numba version", allow_module_level=True)


def test_cupy_interop():
    cupy = Cupy.instance()
    numpy = Numpy.instance()

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
