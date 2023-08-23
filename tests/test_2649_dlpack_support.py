# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._connect.dlpack import DLPackDevice


def test_from_dlpack_numpy():
    np_array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    array = ak.from_dlpack(np_array)
    np_from_ak = ak.to_numpy(array)
    assert np.shares_memory(np_array, np_from_ak)


def test_to_dlpack_numpy():
    np_array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    array = ak.from_numpy(np_array, regulararray=True)
    np_from_ak = np.from_dlpack(array)
    assert np.shares_memory(np_array, np_from_ak)
    assert array.__dlpack_device__()[0] == DLPackDevice.CPU


def test_from_dlpack_cupy():
    # This test only checks cupy usage, it doesn't explicitly test GPU & CPU
    cp = pytest.importorskip("cupy")
    cp_array = cp.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    array = ak.from_dlpack(cp_array)
    cp_from_ak = ak.to_cupy(array)
    assert cp.shares_memory(cp_array, cp_from_ak)


def test_to_dlpack_cupy():
    # This test only checks cupy usage, it doesn't explicitly test GPU & CPU
    cp = pytest.importorskip("cupy")
    cp_array = cp.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    array = ak.from_cupy(cp_array, regulararray=True)
    cp_from_ak = cp.from_dlpack(array)
    assert cp.shares_memory(cp_array, cp_from_ak)


class DLPackOf:
    def __init__(self, array):
        self._array = array

    def __dlpack__(self, stream=None):
        if stream is None:
            return self._array.__dlpack__()
        else:
            return self._array.__dlpack__(stream)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


def test_to_layout():
    np_array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    dlpack_array = DLPackOf(np_array)
    layout = ak.to_layout(dlpack_array)
    assert layout.is_numpy

    np_from_ak = ak.to_numpy(layout)
    assert np.shares_memory(np_array, np_from_ak)


def test_invalid_argument():
    with pytest.raises(
        TypeError, match=r"Expected an object that implements the DLPack protocol"
    ):
        ak.from_dlpack([1, 2, 3])
