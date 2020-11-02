# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    one = awkward1.Array(["uno", "dos", "tres"])
    two = awkward1.Array(["un", "deux", "trois", "quatre"])
    three = awkward1.Array(["onay", "ootay", "eethray"])
    merged = awkward1.concatenate([one, two, three])
    assert awkward1.to_list(merged) == ["uno", "dos", "tres", "un", "deux", "trois", "quatre", "onay", "ootay", "eethray"]
    assert awkward1.to_list(merged == "uno") == [True, False, False, False, False, False, False, False, False, False]
    assert awkward1.to_list(one == numpy.array(["UNO", "dos", "tres"])) == [False, True, True]
    assert awkward1.to_list(merged == numpy.array(["UNO", "dos", "tres", "one", "two", "three", "quatre", "onay", "two", "three"])) == [False, True, True, False, False, False, True, True, False, False]


def test_fromnumpy():
    assert awkward1.to_list(awkward1.from_numpy(numpy.array(["uno", "dos", "tres", "quatro"]))) == ["uno", "dos", "tres", "quatro"]
    assert awkward1.to_list(awkward1.from_numpy(numpy.array([["uno", "dos"], ["tres", "quatro"]]))) == [["uno", "dos"], ["tres", "quatro"]]
    assert awkward1.to_list(awkward1.from_numpy(numpy.array([["uno", "dos"], ["tres", "quatro"]]), regulararray=True)) == [["uno", "dos"], ["tres", "quatro"]]
