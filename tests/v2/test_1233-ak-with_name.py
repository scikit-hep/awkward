# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_with_name():
    array = ak._v2.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ]
    )

    one = ak._v2.operations.with_name(array, "Wilbur")
    assert isinstance(one.layout, ak._v2.contents.Content)
    assert one.layout.parameters["__record__"] == "Wilbur"

    array2 = ak._v2.operations.from_iter(
        [
            [[1], 2.2, [2, 2], 3.3, [3, 3, 3], 4.4, [4, 4, 4, 4]],
            [
                {"x": 0.0, "y": []},
                {"x": 1.1, "y": [1]},
                {"x": 2.2, "y": [2, 2]},
                {"x": 3.3, "y": [3, 3, 3]},
            ],
        ],
        highlevel=False,
    )
    one = ak._v2.operations.with_name(array2, "Wilbur")
    assert one.layout.content.contents[2].parameters["__record__"] == "Wilbur"

    array = ak._v2.highlevel.Array(
        [
            {"a": [[0.0, 4.5], [], None], "b": []},
            {"a": 1.1, "b": [[1]]},
        ]
    )
    one = ak._v2.operations.with_name(array, "James")
    assert isinstance(one.layout, ak._v2.contents.Content)
    assert one.layout.parameters["__record__"] == "James"


def test_simplify_unionarray_with_name():
    one = ak._v2.operations.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak._v2.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak._v2.operations.from_iter(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    tags2 = ak._v2.index.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak._v2.index.Index32(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int32))
    inner = ak._v2.contents.UnionArray(tags2, index2, [two, three])
    tags1 = ak._v2.index.Index8(
        np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    )
    index1 = ak._v2.index.Index64(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int64)
    )
    outer = ak._v2.contents.UnionArray(tags1, index1, [one, inner])
    one = ak._v2.operations.with_name(outer, "James")

    assert outer.contents[1].is_UnionType != one.layout.contents[1].is_UnionType
