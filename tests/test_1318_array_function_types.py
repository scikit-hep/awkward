# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import numpy as np

import awkward as ak


def test_tuple():
    data = (np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64))
    result = np.ravel_multi_index(
        (ak.from_numpy(data[0]), ak.from_numpy(data[1])), (2, 2)
    )
    assert isinstance(result, ak.Array)
    assert ak.almost_equal(result, np.ravel_multi_index(data, (2, 2)))


def test_list():
    A = np.eye(2) * 2
    B = np.eye(3) * 3
    result = np.block(
        [
            [ak.from_numpy(A), ak.from_numpy(np.zeros((2, 3)))],
            [ak.from_numpy(np.ones((3, 2))), ak.from_numpy(B)],
        ]
    )
    assert isinstance(result, ak.Array)
    assert ak.almost_equal(
        result, np.block([[A, np.zeros((2, 3))], [np.ones((3, 2)), B]])
    )


def test_array():
    haystack = np.array([1, 2, 3, 4, 4, 5, 6, 7], dtype=np.int64)
    needle = np.array([5, 0, 2], dtype=np.int64)
    result = np.searchsorted(ak.from_numpy(haystack), ak.from_numpy(needle))
    assert isinstance(result, ak.Array)
    assert ak.almost_equal(result, np.searchsorted(haystack, needle))


def test_scalar():
    data = np.array([1, 2, 3, 4, 3, 2, 1, 2], dtype=np.int64)
    result = np.partition(ak.from_numpy(data), 4)
    assert isinstance(result, ak.Array)
    assert ak.almost_equal(result, np.partition(data, 4))


def test_tuple_of_array():
    data = (
        np.array([5, 4, 3, 2, 1], dtype=np.int64),
        np.array([1, 2, 3, 5, 0], dtype=np.int64),
    )
    result = np.lexsort((ak.from_numpy(data[0]), ak.from_numpy(data[1])))
    assert isinstance(result, ak.Array)
    assert ak.almost_equal(result, np.lexsort(data))
