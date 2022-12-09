# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    arr1 = ak.highlevel.Array({"a": [1, 2], "b": [1, None]})
    arr2 = ak.operations.mask(arr1, [True, True])
    assert isinstance(arr2.layout, ak.contents.ByteMaskedArray)
    assert isinstance(arr2.layout.content, ak.contents.RecordArray)
    assert isinstance(arr2.layout.content["b"], ak.contents.IndexedOptionArray)

    assert isinstance(arr2.b.layout, ak.contents.IndexedOptionArray)
    assert isinstance(arr2.b.layout.content, ak.contents.NumpyArray)

    assert ak.operations.is_none(arr2.b).to_list() == [False, True]
