# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import numpy as np

import awkward as ak


def test_tuple():
    result = np.ravel_multi_index((ak.Array([0, 1]), ak.Array([1, 0])), (2, 2))
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(
        result, np.ravel_multi_index((np.array([0, 1]), np.array([1, 0])), (2, 2))
    )


def test_list():
    result = np.partition(ak.Array([1, 2, 3, 4, 3, 2, 1, 2]), [4, 6])
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(
        result, np.partition(np.array([1, 2, 3, 4, 3, 2, 1, 2]), [4, 6])
    )


def test_scalar():
    result = np.partition(ak.Array([1, 2, 3, 4, 3, 2, 1, 2]), 4)
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(
        result, np.partition(np.array([1, 2, 3, 4, 3, 2, 1, 2]), 4)
    )


def test_array():
    result = np.partition(ak.Array([1, 2, 3, 4, 3, 2, 1, 2]), ak.Array([4, 6]))
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(
        result, np.partition(np.array([1, 2, 3, 4, 3, 2, 1, 2]), np.array([4, 6]))
    )


def test_tuple_of_array():
    result = np.lexsort((ak.Array([5, 4, 3, 2, 1]), ak.Array([1, 2, 3, 5, 0])))
    assert isinstance(result, ak.Array)
    assert ak._util.arrays_approx_equal(
        result,
        np.lexsort((np.array([5, 4, 3, 2, 1]), np.array([1, 2, 3, 5, 0]))),
    )
