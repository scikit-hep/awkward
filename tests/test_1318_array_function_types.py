# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import numpy as np

import awkward as ak


def test_tuple():
    data = (np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64))
    result = np.ravel_multi_index(
        (ak.from_numpy(data[0]), ak.from_numpy(data[1])), (2, 2)
    )
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(result, np.ravel_multi_index(data, (2, 2)))


def test_list():
    data = np.array([1, 2, 3, 4, 3, 2, 1, 2], dtype=np.int64)
    result = np.partition(ak.from_numpy(data), [4, 6])
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(result, np.partition(data, [4, 6]))


def test_array():
    data = np.array([1, 2, 3, 4, 3, 2, 1, 2], dtype=np.int64)
    result = np.partition(ak.from_numpy(data), ak.Array([4, 6]))
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(result, np.partition(data, np.array([4, 6])))


def test_scalar():
    data = np.array([1, 2, 3, 4, 3, 2, 1, 2], dtype=np.int64)
    result = np.partition(ak.from_numpy(data), 4)
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(result, np.partition(data, 4))


def test_tuple_of_array():
    data = (
        np.array([5, 4, 3, 2, 1], dtype=np.int64),
        np.array([1, 2, 3, 5, 0], dtype=np.int64),
    )
    result = np.lexsort((ak.from_numpy(data[0]), ak.from_numpy(data[1])))
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(result, np.lexsort(data))
