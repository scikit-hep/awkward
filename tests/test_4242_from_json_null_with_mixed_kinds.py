# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize(
    ("text", "expected_type", "expected_list"),
    [
        ('[1,"a",null]', "3 * union[?int64, ?string]", [1, "a", None]),
        (
            '[[{}],["",null]]',
            "2 * var * union[?{}, ?string]",
            [[{}], ["", None]],
        ),
        (
            '["a",["b"],null,{}]',
            "4 * union[?string, option[var * string], ?{}]",
            ["a", ["b"], None, {}],
        ),
    ],
)
def test_null_with_mixed_kinds(text, expected_type, expected_list):
    array = ak.from_json(text)
    assert str(array.type) == expected_type
    assert array.tolist() == expected_list


def test_line_delimited():
    array = ak.from_json('1\n"a"\nnull', line_delimited=True)
    assert str(array.type) == "3 * union[?int64, ?string]"
    assert array.tolist() == [1, "a", None]


def test_union_of_options_from_to_json():
    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.array([0, 1, 0], np.int8)),
            ak.index.Index64(np.array([0, 0, 1])),
            [
                ak.contents.IndexedOptionArray(
                    ak.index.Index64(np.array([0, -1])),
                    ak.contents.NumpyArray(np.array([1], np.int64)),
                ),
                ak.contents.IndexedOptionArray(
                    ak.index.Index64(np.array([0])), ak.to_layout(["a"])
                ),
            ],
        )
    )
    text = ak.to_json(array)
    assert text == '[1,"a",null]'
    returned = ak.from_json(text)
    assert str(returned.type) == "3 * union[?int64, ?string]"
    assert returned.tolist() == [1, "a", None]
