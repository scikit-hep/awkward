# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    f = ak.forms.Form.fromjson(
        json.dumps(
            {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "d",
                "primitive": "float64",
                "parameters": {"thing": 1.23},
            }
        )
    )
    assert json.loads(repr(f)) == {
        "class": "NumpyArray",
        "itemsize": 8,
        "format": "d",
        "primitive": "float64",
        "parameters": {"thing": 1.23},  # not 1 (int)
    }
