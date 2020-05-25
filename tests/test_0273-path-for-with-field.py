# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_one_level():
    base = awkward1.zip({"a" : awkward1.zip({"x" : [1, 2, 3]}), "b" : [1, 2, 3]}, depth_limit=1)
    what = awkward1.Array([1.1, 2.2, 3.3], check_valid=True)
    assert awkward1.to_list(awkward1.with_field(base, what, where=["a", "y"])) == [{'b': 1, 'a': {'x': 1, 'y': 1.1}}, {'b': 2, 'a': {'x': 2, 'y': 2.2}}, {'b': 3, 'a': {'x': 3, 'y': 3.3}}]

    base["a", "y"] = what
    assert awkward1.to_list(base) == [{'b': 1, 'a': {'x': 1, 'y': 1.1}}, {'b': 2, 'a': {'x': 2, 'y': 2.2}}, {'b': 3, 'a': {'x': 3, 'y': 3.3}}]

def test_two_level():
    base = awkward1.zip({"A" : awkward1.zip({"a" : awkward1.zip({"x" : [1, 2, 3]}), "b" : [1, 2, 3]}), "B" : [1, 2, 3]}, depth_limit=1)
    what = awkward1.Array([1.1, 2.2, 3.3], check_valid=True)
    assert awkward1.to_list(awkward1.with_field(base, what, where=["A", "a", "y"])) == [{'B': 1, 'A': {'b': 1, 'a': {'x': 1, 'y': 1.1}}}, {'B': 2, 'A': {'b': 2, 'a': {'x': 2, 'y': 2.2}}}, {'B': 3, 'A': {'b': 3, 'a': {'x': 3, 'y': 3.3}}}]

    base["A", "a", "y"] = what
    assert awkward1.to_list(base) == [{'B': 1, 'A': {'b': 1, 'a': {'x': 1, 'y': 1.1}}}, {'B': 2, 'A': {'b': 2, 'a': {'x': 2, 'y': 2.2}}}, {'B': 3, 'A': {'b': 3, 'a': {'x': 3, 'y': 3.3}}}]

def test_replace_the_only_field():
    base = awkward1.zip({"a" : awkward1.zip({"x" : [1, 2, 3]})}, depth_limit=1)
    what = awkward1.Array([1.1, 2.2, 3.3], check_valid=True)
    assert awkward1.to_list(awkward1.with_field(base, what, where=["a", "y"])) == [{'a': {'x': 1, 'y': 1.1}}, {'a': {'x': 2, 'y': 2.2}}, {'a': {'x': 3, 'y': 3.3}}]

    base["a", "y"] = what
    assert awkward1.to_list(base) == [{'a': {'x': 1, 'y': 1.1}}, {'a': {'x': 2, 'y': 2.2}}, {'a': {'x': 3, 'y': 3.3}}]
