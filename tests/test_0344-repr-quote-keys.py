# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

@pytest.mark.skipif(ak._util.py27 and ak._util.win, reason="Windows Python 2.7 inserts 'L' after numbers in repr")
def test():
    assert str(ak.Array([
        {"a": 1, "a b": 2}, {"a": 1, "a b": 2}, {"a": 1, "a b": 2}, {"a": 1, "a b": 2}
    ])) in (
        "[{a: 1, 'a b': 2}, {a: 1, 'a b': 2}, {a: 1, 'a b': 2}, {a: 1, 'a b': 2}]",
        "[{'a b': 2, a: 1}, {'a b': 2, a: 1}, {'a b': 2, a: 1}, {'a b': 2, a: 1}]"
    )
    assert repr(ak.Array([
        {"a": 1, "a b": 2}, {"a": 1, "a b": 2}, {"a": 1, "a b": 2}
    ])) in (
        "<Array [{a: 1, 'a b': 2}, ... {a: 1, 'a b': 2}] type='3 * {\"a\": int64, \"a b\": in...'>"
        "<Array [{'a b': 2, a: 1}, ... {'a b': 2, a: 1}] type='3 * {\"a b\": int64, \"a\": in...'>"
    )
