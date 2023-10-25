# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from datetime import datetime

import numpy as np
import pytest

import awkward as ak


def test_strings():
    assert ak.to_layout("hello").is_equal_to(
        ak.contents.NumpyArray(
            np.array([104, 101, 108, 108, 111], dtype=np.uint8),
            parameters={"__array__": "char"},
        )
    )
    assert ak.to_layout("hello", string_policy="promote").is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 5]),
            ak.contents.NumpyArray(
                np.array([104, 101, 108, 108, 111], dtype=np.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )
    )
    assert ak.to_layout("hello", string_policy="pass-through") == "hello"
    with pytest.raises(TypeError, match=r"string conversion/promotion is disabled"):
        ak.to_layout("hello", string_policy="error")


def test_primitives():
    with pytest.raises(TypeError, match=r"scalar conversion/promotion is disabled"):
        ak.to_layout(1j, primitive_policy="error")
    with pytest.raises(TypeError, match=r"scalar conversion/promotion is disabled"):
        ak.to_layout(False, primitive_policy="error")
    with pytest.raises(TypeError, match=r"scalar conversion/promotion is disabled"):
        ak.to_layout(datetime.now(), primitive_policy="error")

    with pytest.raises(TypeError, match=r"scalar conversion/promotion is disabled"):
        ak.to_layout(1, primitive_policy="error")
    assert ak.to_layout(1).is_equal_to(
        ak.contents.NumpyArray(np.array([1], dtype=np.int64))
    )
    out = ak.to_layout(1, primitive_policy="pass-through")
    assert isinstance(out, int) and out == 1

    with pytest.raises(TypeError, match=r"scalar conversion/promotion is disabled"):
        ak.to_layout(4.0, primitive_policy="error")
    assert ak.to_layout(4.0).is_equal_to(
        ak.contents.NumpyArray(np.array([4.0], dtype=np.float64))
    )
    out = ak.to_layout(4.0, primitive_policy="pass-through")
    assert isinstance(out, float) and out == 4.0
