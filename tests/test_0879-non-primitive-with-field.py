# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_unknown_type():
    array = ak.Array({"x": np.arange(10)})
    array = ak.operations.with_field(array=array, what=None, where="unknown field1")
    array = ak.operations.with_field(array=array, what=[None], where="unknown field2")

    # Try to access the type of a single element
    # This raises a ValueError in #879
    tpe1 = array["unknown field1"].type
    tpe2 = array["unknown field2"].type
    assert str(tpe1) == "10 * ?unknown"
    assert str(tpe2) == "10 * ?unknown"


def test_in_place_wrapper_broadcasting():
    array = ak.Array({"x": np.arange(3)})
    array["unknown field"] = None

    assert array["unknown field"].to_list() == [None, None, None]
    assert ak.operations.fields(array) == ["x", "unknown field"]
