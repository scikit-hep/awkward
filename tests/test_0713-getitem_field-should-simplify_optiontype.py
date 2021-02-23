# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    arr1 = ak.Array({"a": [1, 2], "b": [1, None]})
    arr2 = ak.mask(arr1, [True, True])
    assert isinstance(arr2.layout, ak.layout.ByteMaskedArray)
    assert isinstance(arr2.layout.content, ak.layout.RecordArray)
    assert isinstance(arr2.layout.content.field("b"), ak.layout.IndexedOptionArray64)

    assert isinstance(arr2.b.layout, ak.layout.IndexedOptionArray64)
    assert isinstance(arr2.b.layout.content, ak.layout.NumpyArray)

    assert ak.is_none(arr2.b).tolist() == [False, True]

    arr3 = ak.virtual(lambda: arr2, form=arr2.layout.form, length=len(arr2))
    assert ak.is_none(arr3.b).tolist() == [False, True]
