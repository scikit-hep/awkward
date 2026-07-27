# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test():
    array = ak.Array(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))
    assert ak.array_equal(array, array)

    array1 = ak.Array(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))
    array2 = ak.Array(np.array(["2020-01-03", "2020-01-04"], dtype="datetime64[D]"))
    assert not ak.array_equal(array1, array2)

    array1 = ak.Array(np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"))
    array2 = ak.Array(np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"))
    assert not ak.array_equal(array1, array2, equal_nan=False)
    assert ak.array_equal(array1, array2, equal_nan=True)
