# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


class ReversibleArray(ak.Array):
    def reversed(self):
        return self[..., ::-1]


def test_class():
    behavior = {"reversible": ReversibleArray}
    reversible_array = ak.with_parameter(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]], "__list__", "reversible", behavior=behavior
    )
    assert isinstance(reversible_array, ReversibleArray)
    assert reversible_array.to_list() == [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert reversible_array.reversed().to_list() == [[3, 2, 1], [7, 6, 5, 4], [9, 8]]


def test_deep_class():
    behavior = {"reversible": ReversibleArray, ("*", "reversible"): ReversibleArray}
    reversible_array = ak.with_parameter(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]], "__list__", "reversible", behavior=behavior
    )
    outer_array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 2, 3, 3]), reversible_array.layout
        ),
        behavior=behavior,
    )
    assert isinstance(outer_array, ReversibleArray)
    assert outer_array.to_list() == [[[1, 2, 3], [4, 5, 6, 7]], [[8, 9]], []]
    assert outer_array.reversed().to_list() == [
        [[3, 2, 1], [7, 6, 5, 4]],
        [[9, 8]],
        [],
    ]


def test_ufunc():
    behavior = {"reversible": ReversibleArray}
    reversible_array = ak.with_parameter(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]], "__list__", "reversible", behavior=behavior
    )
    assert isinstance(reversible_array, ReversibleArray)

    def reversible_add(x, y):
        return ak.with_parameter(x.reversed(), "__list__", None) + ak.with_parameter(
            y.reversed(), "__list__", None
        )

    ak.behavior[np.add, "reversible", "reversible"] = reversible_add

    assert (reversible_array + reversible_array).to_list() == [
        [6, 4, 2],
        [14, 12, 10, 8],
        [18, 16],
    ]
    with pytest.raises(TypeError, match=r"overloads for custom types"):
        reversible_array + ak.with_parameter(
            reversible_array, "__list__", "non-reversible"
        )

    # TODO: this should become true once string broadcasting is addressed
    #       so that we can generalise the solution
    # # We can't apply ufunc to types without overloads
    # with pytest.raises(TypeError, match=r"overloads for custom types"):
    #     reversible_array + ak.without_parameters(reversible_array)


def test_string_ufuncs():
    # Strings can't gt by default
    with pytest.raises(TypeError, match=r"is not implemented for string types"):
        np.greater(
            ak.Array(["do", "not", "go", "gentle"]),
            ak.Array(["gentle", "into", "that", "good"]),
        )

    # Implement gt for a custom string type
    def string_greater(left, right):
        return ak.num(left) > ak.num(right)

    behavior = {(np.greater, "stringy", "stringy"): string_greater}
    result = np.greater(
        ak.with_parameter(
            ["do", "not", "go", "gentle"], "__list__", "stringy", behavior=behavior
        ),
        ak.with_parameter(
            ["gentle", "into", "that", "good"], "__list__", "stringy", behavior=behavior
        ),
    )
    assert result.to_list() == [False, False, False, True]


def test_string_class():
    ak.behavior["reversible-string"] = ReversibleArray

    strings = ak.with_parameter(["hi", "book", "cats"], "__list__", "reversible-string")
    assert isinstance(strings, ReversibleArray)
    assert strings.to_list() == ["hi", "book", "cats"]
    assert strings.reversed().to_list() == ["cats", "book", "hi"]
