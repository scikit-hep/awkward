# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_with_name():
    array = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ]
    )

    one = ak.operations.with_name(array, "Wilbur")
    assert isinstance(one.layout, ak.contents.Content)
    assert one.layout.parameters["__record__"] == "Wilbur"

    array2 = ak.operations.from_iter(
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
    one = ak.operations.with_name(array2, "Wilbur")
    assert one.layout.content.contents[2].parameters["__record__"] == "Wilbur"

    array = ak.highlevel.Array(
        [
            {"a": [[0.0, 4.5], [], None], "b": []},
            {"a": 1.1, "b": [[1]]},
        ]
    )
    one = ak.operations.with_name(array, "James")
    assert isinstance(one.layout, ak.contents.Content)
    assert one.layout.parameters["__record__"] == "James"


def test_simplify_unionarray_with_name():
    one = ak.operations.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.operations.from_iter(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    tags2 = ak.index.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.index.Index32(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int32))
    inner = ak.contents.UnionArray(tags2, index2, [two, three])
    tags1 = ak.index.Index8(
        np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    )
    index1 = ak.index.Index64(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int64)
    )
    outer = ak.contents.UnionArray.simplified(tags1, index1, [one, inner])
    one = ak.operations.with_name(outer, "James")

    assert outer.to_list() == [
        5,
        4,
        [],
        {"x": 0.0, "y": []},
        [1],
        3,
        {"x": 1.1, "y": [1]},
        [2, 2],
        [3, 3, 3],
        2,
        {"x": 2.2, "y": [2, 2]},
        1,
    ]
