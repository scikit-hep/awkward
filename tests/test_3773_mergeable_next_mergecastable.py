# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_can_cast():
    numpy_like = ak._nplikes.numpy.Numpy.instance()
    # default casting is "same_kind"
    assert numpy_like.can_cast(np.float32, np.float64)
    assert numpy_like.can_cast(np.int32, np.int64)
    assert numpy_like.can_cast(np.float64, np.float32)
    assert numpy_like.can_cast(np.int64, np.int32)

    assert numpy_like.can_cast(np.float32, np.float64, casting="safe")
    assert numpy_like.can_cast(np.int32, np.int64, casting="safe")
    assert not numpy_like.can_cast(np.float64, np.float32, casting="safe")
    assert not numpy_like.can_cast(np.int64, np.int32, casting="safe")

    assert numpy_like.can_cast(np.float32, np.float32, casting="equiv")
    assert not numpy_like.can_cast(np.float32, np.float64, casting="equiv")
    assert not numpy_like.can_cast(np.int32, np.int64, casting="equiv")
    assert not numpy_like.can_cast(np.float64, np.float32, casting="equiv")
    assert not numpy_like.can_cast(np.int64, np.int32, casting="equiv")


def test_mergeable_next_mergecastable():
    array1 = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32))
    array2 = ak.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64))
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="same_kind")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="family")
    assert not array1._mergeable_next(array2, mergebool=False, mergecastable="equiv")

    array1 = ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    array2 = ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64))
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="same_kind")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="family")
    assert not array1._mergeable_next(array2, mergebool=False, mergecastable="equiv")

    array1 = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32))
    array2 = ak.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int32))
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="same_kind")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="family")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="equiv")

    array1 = ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    array2 = ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="same_kind")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="family")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="equiv")

    array1 = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    array2 = ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0], dtype=np.float64))
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="same_kind")
    assert not array1._mergeable_next(array2, mergebool=False, mergecastable="family")
    assert not array1._mergeable_next(array2, mergebool=False, mergecastable="equiv")

    array1 = ak.contents.NumpyArray(np.array([True, False, True], dtype=np.bool_))
    array2 = ak.contents.NumpyArray(np.array([False, True, False], dtype=np.bool_))
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="same_kind")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="family")
    assert array1._mergeable_next(array2, mergebool=False, mergecastable="equiv")
    assert array1._mergeable_next(array2, mergebool=True, mergecastable="same_kind")
    assert array1._mergeable_next(array2, mergebool=True, mergecastable="family")
    assert array1._mergeable_next(array2, mergebool=True, mergecastable="equiv")
