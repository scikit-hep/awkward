# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_array():
    @numba.njit
    def f1(x):
        return np.array(x)

    assert isinstance(
        f1(ak.highlevel.Array([[[1], [2], [3]], [[4], [5], [6]]])), np.ndarray
    )
    assert f1(ak.highlevel.Array([[[1], [2], [3]], [[4], [5], [6]]])).tolist() == [
        [[1], [2], [3]],
        [[4], [5], [6]],
    ]
    assert f1(ak.highlevel.Array([[1, 2, 3], [4, 5, 6]])).tolist() == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert f1(ak.highlevel.Array([1, 2, 3, 4, 5, 6])).tolist() == [1, 2, 3, 4, 5, 6]

    with pytest.raises(ValueError):
        f1(ak.highlevel.Array([[1, 2, 3, 4], [5, 6]]))


def test_asarray():
    @numba.njit
    def f1(x):
        return np.asarray(x[-1][1:])

    akarray = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]
    )
    nparray = f1(akarray)

    assert nparray.tolist() == [7.7, 8.8, 9.9]
