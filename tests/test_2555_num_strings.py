# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_string_axis_0():
    array = ak.Array(["abc", "de", ""])
    assert ak.num(array, axis=0) == 3


def test_string_inner_axis_raises():
    array = ak.Array(["abc", "de", ""])
    with pytest.raises(np.exceptions.AxisError, match="strings"):
        ak.num(array, axis=1)
    # strings are leaves, so the innermost axis is axis=0
    assert ak.num(array, axis=-1) == 3


def test_bytestring_inner_axis_raises():
    array = ak.Array([b"abc", b"de"])
    assert ak.num(array, axis=0) == 2
    with pytest.raises(np.exceptions.AxisError, match="strings"):
        ak.num(array, axis=1)


def test_list_of_strings():
    array = ak.Array([["abc", "de"], [], ["f"]])
    assert ak.num(array, axis=0) == 3
    assert ak.num(array, axis=1).to_list() == [2, 0, 1]
    assert ak.num(array, axis=-1).to_list() == [2, 0, 1]
    with pytest.raises(np.exceptions.AxisError, match="strings"):
        ak.num(array, axis=2)


def test_record_with_string_field():
    array = ak.Array([{"x": "abc", "y": [1, 2]}, {"x": "d", "y": []}])
    with pytest.raises(np.exceptions.AxisError, match="strings"):
        ak.num(array, axis=1)


def test_without_parameters_counts_characters():
    array = ak.Array(["abc", "de"])
    assert ak.num(ak.without_parameters(array), axis=1).to_list() == [3, 2]


def test_typetracer():
    array = ak.Array(["abc", "de"], backend="typetracer")
    with pytest.raises(np.exceptions.AxisError, match="strings"):
        ak.num(array, axis=1)
