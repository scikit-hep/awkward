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
