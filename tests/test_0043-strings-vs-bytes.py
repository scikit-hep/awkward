# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([{"x": "one"}, {"x": "two"}, {"x": "three"}], check_valid=True)
    assert ak.to_list(array) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert ak.to_list(ak.from_iter(ak.to_list(array))) == [
        {"x": "one"},
        {"x": "two"},
        {"x": "three"},
    ]
    assert ak.to_list(array.layout) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert ak.to_list(ak.from_iter(ak.to_list(array.layout))) == [
        {"x": "one"},
        {"x": "two"},
        {"x": "three"},
    ]
