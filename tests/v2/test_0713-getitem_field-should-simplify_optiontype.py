# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():
    arr1 = ak._v2.highlevel.Array({"a": [1, 2], "b": [1, None]})
    arr2 = ak._v2.operations.mask(arr1, [True, True])
    assert isinstance(arr2.layout, ak._v2.contents.ByteMaskedArray)
    assert isinstance(arr2.layout.content, ak._v2.contents.RecordArray)
    assert isinstance(arr2.layout.content["b"], ak._v2.contents.IndexedOptionArray)

    assert isinstance(arr2.b.layout, ak._v2.contents.IndexedOptionArray)
    assert isinstance(arr2.b.layout.content, ak._v2.contents.NumpyArray)

    assert ak._v2.operations.is_none(arr2.b).tolist() == [False, True]
