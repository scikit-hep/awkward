# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest

import numpy
import awkward1


def test():
    example = awkward1.Array([[{"x": 1, "y": []}, {"x": 2, "y": [1, 1]}], []])
    cache = {}
    virtualarray = awkward1.virtual(lambda: example, form=example.layout.form, length=len(example))
    assert len(cache) == 0

    tmp1 = virtualarray["x"]
    assert json.loads(str(tmp1.layout.form)) == {
        "class": "VirtualArray",
        "form": {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": "int64"
        },
        "has_length": True
    }
    assert len(cache) == 0

    tmp2 = virtualarray["y"]
    assert json.loads(str(tmp2.layout.form)) == {
        "class": "VirtualArray",
        "form": {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": {
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": "int64"
            }
        },
        "has_length": True
    }
    assert len(cache) == 0

    assert tmp1.tolist() == [[1, 2], []]
    assert tmp2.tolist() == [[[], [1, 1]], []]

def test_no_break_regular_broadcasting():
    # because we redefined Form::has_virtual_form to agree with an interpretation of its name in English

    assert (numpy.array([[1, 2, 3], [4, 5, 6]]) + numpy.array([[10], [20]])).tolist() == [[11, 12, 13], [24, 25, 26]]
    assert (awkward1.Array(numpy.array([[1, 2, 3], [4, 5, 6]])) + awkward1.Array(numpy.array([[10], [20]]))).tolist() == [[11, 12, 13], [24, 25, 26]]
    with pytest.raises(ValueError):
        awkward1.Array([[1, 2, 3], [4, 5, 6]]) + awkward1.Array([[10], [20]])
    left, right = awkward1.Array(numpy.array([[1, 2, 3], [4, 5, 6]])), awkward1.Array(numpy.array([[10], [20]]))
    assert (awkward1.virtual(lambda: left, form=left.layout.form, length=len(left)) + awkward1.virtual(lambda: right, form=right.layout.form, length=len(right))).tolist() == [[11, 12, 13], [24, 25, 26]]
    with pytest.raises(ValueError):
        left, right = awkward1.Array([[1, 2, 3], [4, 5, 6]]), awkward1.Array([[10], [20]])
        awkward1.virtual(lambda: left, form=left.layout.form, length=len(left)) + awkward1.virtual(lambda: right, form=right.layout.form, length=len(right))

    assert (numpy.array([[1, 2, 3], [4, 5, 6]]) + numpy.array([10])).tolist() == [[11, 12, 13], [14, 15, 16]]
    assert (awkward1.Array(numpy.array([[1, 2, 3], [4, 5, 6]])) + awkward1.Array(numpy.array([10]))).tolist() == [[11, 12, 13], [14, 15, 16]]
    assert (awkward1.Array([[1, 2, 3], [4, 5, 6]]) + awkward1.Array([10])).tolist() == [[11, 12, 13], [14, 15, 16]]
    left, right = awkward1.Array(numpy.array([[1, 2, 3], [4, 5, 6]])), awkward1.Array(numpy.array([10]))
    assert (awkward1.virtual(lambda: left, form=left.layout.form, length=len(left)) + awkward1.virtual(lambda: right, form=right.layout.form, length=len(right))).tolist() == [[11, 12, 13], [14, 15, 16]]
    left, right = awkward1.Array([[1, 2, 3], [4, 5, 6]]), awkward1.Array([10])
    assert (awkward1.virtual(lambda: left, form=left.layout.form, length=len(left)) + awkward1.virtual(lambda: right, form=right.layout.form, length=len(right))).tolist() == [[11, 12, 13], [14, 15, 16]]

    assert (awkward1.Array([[1, 2, 3], [4, 5, 6]]) + awkward1.Array([10, 20])).tolist() == [[11, 12, 13], [24, 25, 26]]
    left, right = awkward1.Array([[1, 2, 3], [4, 5, 6]]), awkward1.Array([10, 20])
    assert (awkward1.virtual(lambda: left, form=left.layout.form, length=len(left)) + awkward1.virtual(lambda: right, form=right.layout.form, length=len(right))).tolist() == [[11, 12, 13], [24, 25, 26]]
