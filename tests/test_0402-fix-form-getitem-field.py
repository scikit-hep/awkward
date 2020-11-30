# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import json

import pytest
import numpy as np
import awkward1 as ak


def test():
    example = ak.Array([[{"x": 1, "y": []}, {"x": 2, "y": [1, 1]}], []])
    cache = {}
    virtualarray = ak.virtual(
        lambda: example, form=example.layout.form, length=len(example)
    )
    assert len(cache) == 0

    tmp1 = virtualarray["x"]
    assert json.loads(str(tmp1.layout.form)) == {
        "class": "VirtualArray",
        "form": {"class": "ListOffsetArray64", "offsets": "i64", "content": "int64"},
        "has_length": True,
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
                "content": "int64",
            },
        },
        "has_length": True,
    }
    assert len(cache) == 0

    assert tmp1.tolist() == [[1, 2], []]
    assert tmp2.tolist() == [[[], [1, 1]], []]


def test_no_break_regular_broadcasting():
    # because we redefined Form::has_virtual_form to agree with an interpretation of its name in English

    assert (np.array([[1, 2, 3], [4, 5, 6]]) + np.array([[10], [20]])).tolist() == [
        [11, 12, 13],
        [24, 25, 26],
    ]
    assert (
        ak.Array(np.array([[1, 2, 3], [4, 5, 6]])) + ak.Array(np.array([[10], [20]]))
    ).tolist() == [[11, 12, 13], [24, 25, 26]]
    with pytest.raises(ValueError):
        ak.Array([[1, 2, 3], [4, 5, 6]]) + ak.Array([[10], [20]])
    left, right = (
        ak.Array(np.array([[1, 2, 3], [4, 5, 6]])),
        ak.Array(np.array([[10], [20]])),
    )
    assert (
        ak.virtual(lambda: left, form=left.layout.form, length=len(left))
        + ak.virtual(lambda: right, form=right.layout.form, length=len(right))
    ).tolist() == [[11, 12, 13], [24, 25, 26]]
    with pytest.raises(ValueError):
        left, right = ak.Array([[1, 2, 3], [4, 5, 6]]), ak.Array([[10], [20]])
        ak.virtual(lambda: left, form=left.layout.form, length=len(left)) + ak.virtual(
            lambda: right, form=right.layout.form, length=len(right)
        )

    assert (np.array([[1, 2, 3], [4, 5, 6]]) + np.array([10])).tolist() == [
        [11, 12, 13],
        [14, 15, 16],
    ]
    assert (
        ak.Array(np.array([[1, 2, 3], [4, 5, 6]])) + ak.Array(np.array([10]))
    ).tolist() == [[11, 12, 13], [14, 15, 16]]
    assert (ak.Array([[1, 2, 3], [4, 5, 6]]) + ak.Array([10])).tolist() == [
        [11, 12, 13],
        [14, 15, 16],
    ]
    left, right = ak.Array(np.array([[1, 2, 3], [4, 5, 6]])), ak.Array(np.array([10]))
    assert (
        ak.virtual(lambda: left, form=left.layout.form, length=len(left))
        + ak.virtual(lambda: right, form=right.layout.form, length=len(right))
    ).tolist() == [[11, 12, 13], [14, 15, 16]]
    left, right = ak.Array([[1, 2, 3], [4, 5, 6]]), ak.Array([10])
    assert (
        ak.virtual(lambda: left, form=left.layout.form, length=len(left))
        + ak.virtual(lambda: right, form=right.layout.form, length=len(right))
    ).tolist() == [[11, 12, 13], [14, 15, 16]]

    assert (ak.Array([[1, 2, 3], [4, 5, 6]]) + ak.Array([10, 20])).tolist() == [
        [11, 12, 13],
        [24, 25, 26],
    ]
    left, right = ak.Array([[1, 2, 3], [4, 5, 6]]), ak.Array([10, 20])
    assert (
        ak.virtual(lambda: left, form=left.layout.form, length=len(left))
        + ak.virtual(lambda: right, form=right.layout.form, length=len(right))
    ).tolist() == [[11, 12, 13], [24, 25, 26]]
