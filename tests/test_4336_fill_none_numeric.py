# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def indexedoption(index, content, *, dtype=np.int64, parameters=None, inner=None):
    return ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array(index, dtype=np.int64)),
            ak.contents.NumpyArray(np.array(content, dtype=dtype), parameters=inner),
            parameters=parameters,
        )
    )


def test_fast_path_is_not_an_option_or_union():
    array = ak.Array([1.1, None, 3.3, None])
    assert isinstance(array.layout, ak.contents.IndexedOptionArray)

    result = ak.fill_none(array, 0, axis=-1)
    assert result.to_list() == [1.1, 0.0, 3.3, 0.0]
    assert str(result.type) == "4 * float64"
    assert isinstance(result.layout, ak.contents.NumpyArray)


def test_fast_path_inside_jagged_lists():
    array = ak.Array([[1.1, None, 3.3], [], [None, 5.5]])
    result = ak.fill_none(array, 0, axis=-1)
    assert result.to_list() == [[1.1, 0.0, 3.3], [], [0.0, 5.5]]
    assert str(result.type) == "3 * var * float64"


def test_fast_path_promotes_the_fill_value_dtype():
    result = ak.fill_none(indexedoption([0, -1, 1], [1, 2]), 1.5, axis=-1)
    assert result.to_list() == [1.0, 1.5, 2.0]
    assert str(result.type) == "3 * float64"


def test_no_missing_and_all_missing():
    # every entry present: the gather still has to honour the index permutation
    result = ak.fill_none(indexedoption([2, 0, 1], [1, 2, 3]), 0, axis=-1)
    assert result.to_list() == [3, 1, 2]
    assert str(result.type) == "3 * int64"

    # every entry missing, over a non-empty content
    result = ak.fill_none(indexedoption([-1, -1, -1], [1, 2]), 0, axis=-1)
    assert result.to_list() == [0, 0, 0]
    assert str(result.type) == "3 * int64"

    # every entry missing, over an empty content: bails out, because the fast path
    # gathers missing entries at position 0
    result = ak.fill_none(indexedoption([-1, -1], []), 0, axis=-1)
    assert result.to_list() == [0, 0]
    assert str(result.type) == "2 * int64"

    # ... and the same thing built from Python, whose content is an EmptyArray
    result = ak.fill_none(ak.Array([None, None]), 0, axis=-1)
    assert result.to_list() == [0, 0]
    assert str(result.type) == "2 * int64"


def test_parameters():
    # the leaf's own parameters are not shared by the fill value, so they are dropped
    array = indexedoption([0, -1, 1], [1, 2], inner={"foo": "bar"})
    assert str(ak.fill_none(array, 0, axis=-1).type) == "3 * int64"

    # the option-type node's parameters survive onto the result
    array = indexedoption([0, -1, 1], [1, 2], parameters={"outer": "yes"})
    result = ak.fill_none(array, 0, axis=-1)
    assert result.layout.parameters == {"outer": "yes"}
    assert str(result.type) == '3 * int64[parameters={"outer": "yes"}]'


@pytest.mark.parametrize(
    ("array", "value", "axis", "expected", "expected_type"),
    [
        pytest.param(
            ak.Array(["one", None, "three"]),
            "?",
            -1,
            ["one", "?", "three"],
            "3 * string",
            id="string-content",
        ),
        pytest.param(
            ak.Array([{"x": 1}, None, {"x": 3}]),
            {"x": 0},
            0,
            [{"x": 1}, {"x": 0}, {"x": 3}],
            "3 * {x: int64}",
            id="record-content",
        ),
        pytest.param(
            ak.Array([[1, 2], None, [3]]),
            [],
            0,
            [[1, 2], [], [3]],
            "3 * var * int64",
            id="list-content",
        ),
        pytest.param(
            indexedoption([0, -1, 1], ["2021-01-01", "2021-01-02"], dtype="M8[D]"),
            0,
            -1,
            [
                np.datetime64("2021-01-01"),
                0,
                np.datetime64("2021-01-02"),
            ],
            "3 * union[datetime64[D], int64]",
            id="unmergeable-dtype",
        ),
        pytest.param(
            indexedoption(
                [0, -1, 1], [65, 66], dtype=np.uint8, inner={"__array__": "char"}
            ),
            np.uint8(0),
            -1,
            ["A", 0, "B"],
            "3 * union[char, uint8]",
            id="unmergeable-parameters",
        ),
        pytest.param(
            ak.Array([1.1, None, 3.3]),
            ak.Array([None]),
            -1,
            [1.1, [None], 3.3],
            "3 * union[float64, 1 * ?unknown]",
            id="option-fill-value",
        ),
        pytest.param(
            ak.Array([1.1, None, 3.3]),
            [1, 2],
            -1,
            [1.1, [1, 2], 3.3],
            "3 * union[float64, 2 * int64]",
            id="non-scalar-fill-value",
        ),
    ],
)
def test_fall_back_to_the_union_path(array, value, axis, expected, expected_type):
    result = ak.fill_none(array, value, axis=axis)
    assert result.to_list() == expected
    assert str(result.type) == expected_type


def test_multidimensional_content():
    # a multidimensional leaf is not a scalar leaf: the old path handles it
    array = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, -1, 1], dtype=np.int64)),
            ak.contents.NumpyArray(np.arange(6, dtype=np.int64).reshape(2, 3)),
        )
    )
    result = ak.fill_none(array, 0, axis=-1)
    assert result.to_list() == [[0, 1, 2], None, [3, 4, 5]]
    assert str(result.type) == "3 * option[3 * int64]"


def test_categorical_content_is_unchanged():
    # a categorical cannot be flattened into a bare leaf, so ``fill_none`` has always
    # rejected it; this pins that the fast path does not change that
    array = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, 1, -1, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            parameters={"__array__": "categorical"},
        )
    )
    with pytest.raises(TypeError, match="categorical"):
        ak.fill_none(array, 0, axis=-1)


@pytest.mark.parametrize(
    ("array", "value"),
    [
        pytest.param(ak.Array([1.1, None, 3.3]), 0, id="numeric"),
        pytest.param(ak.Array([[1.1, None], [], [None]]), 0, id="jagged"),
        pytest.param(ak.Array(["one", None, "three"]), "?", id="string"),
    ],
)
def test_typetracer_form_matches(array, value):
    concrete = ak.fill_none(array, value, axis=-1).layout.form
    typetracer = ak.fill_none(
        ak.Array(array.layout.to_typetracer(forget_length=True)), value, axis=-1
    ).layout.form
    assert concrete == typetracer
