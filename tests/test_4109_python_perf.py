# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np

import awkward as ak

to_list = ak.operations.to_list


def test_unique_values_still_correct():
    nplike = ak.Array([1]).layout.backend.nplike
    data = np.array([3, 1, 2, 1, 3, 2, 5], dtype=np.int64)
    result = np.asarray(nplike.unique_values(data))
    assert to_list(result) == [1, 2, 3, 5]


def test_is_unique_subranges():
    assert ak._do.is_unique(ak.Array([[1, 2, 3], [1, 2, 3], [4, 5]]).layout) is False
    assert ak._do.is_unique(ak.Array([[1, 2, 3], [4, 5, 6]]).layout) is True


def test_is_unique_non_contiguous_buffer():
    # A non-contiguous NumpyArray must still be handled by subrange_equal.
    base = np.arange(20, dtype=np.int64).reshape(10, 2)
    col = ak.Array(base[:, 0])  # non-contiguous view
    grouped = ak.unflatten(col, [3, 3, 4])
    assert ak._do.is_unique(grouped.layout) is True
