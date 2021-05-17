# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_getitem_field():
    a1 = ak.zip({"a": [[1], [], [2, 3]], "b": [[4], [], [5, 6]]}, with_name="a1")
    a2 = ak.zip({"a": [[7, 8], [9], []], "b": [[10, 11], [12], []]}, with_name="a2")
    union = ak.where([True, False, True], a1, a2)
    assert str(union.a.type) == "3 * var * int64"


def test_flatten_axis_none():
    a1 = ak.zip({"a": [[1], [], [2, 3]], "b": [[4], [], [5, 6]]}, with_name="a1")
    a2 = ak.zip({"a": [[7, 8], [9], []], "b": [[10, 11], [12], []]}, with_name="a2")
    union = ak.where([True, False, True], a1, a2)
    assert set(ak.flatten(union, axis=None)) == set([1, 2, 3, 4, 5, 6, 9, 12])
