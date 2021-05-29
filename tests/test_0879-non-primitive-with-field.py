# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_unknown_type():
    array = ak.Array({"x": np.arange(10)})
    array = ak.with_field(base=array, what=None, where="unknown field1")
    array = ak.with_field(base=array, what=[None], where="unknown field2")

    # Try to access the type of a single element
    # This raises a ValueError in #879
    tpe1 = array["unknown field1"].type
    tpe2 = array["unknown field2"].type
    assert str(tpe1) == "10 * ?unknown"
    assert str(tpe2) == "10 * ?unknown"


def test_in_place_wrapper_broadcasting():
    array = ak.Array({"x": np.arange(3)})
    array["unknown field"] = None

    assert array["unknown field"].tolist() == [None, None, None]
