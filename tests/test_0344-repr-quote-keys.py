# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1


def test():
    assert str(awkward1.Array([
        {"a": 1, "a b": 2}, {"a": 1, "a b": 2}, {"a": 1, "a b": 2}, {"a": 1, "a b": 2}
    ])) in (
        "[{a: 1, 'a b': 2}, {a: 1, 'a b': 2}, {a: 1, 'a b': 2}, {a: 1, 'a b': 2}]",
        "[{'a b': 2, a: 1}, {'a b': 2, a: 1}, {'a b': 2, a: 1}, {'a b': 2, a: 1}]"
    )
    assert repr(awkward1.Array([
        {"a": 1, "a b": 2}, {"a": 1, "a b": 2}, {"a": 1, "a b": 2}
    ])) in (
        "<Array [{a: 1, 'a b': 2}, ... {a: 1, 'a b': 2}] type='3 * {\"a\": int64, \"a b\": in...'>"
        "<Array [{'a b': 2, a: 1}, ... {'a b': 2, a: 1}] type='3 * {\"a b\": int64, \"a\": in...'>"
    )
