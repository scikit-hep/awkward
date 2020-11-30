# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_getitem():
    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))

    assert np.asarray(ak.layout.UnionArray8_32.regular_index(tags)).tolist() == [
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    assert np.asarray(ak.layout.UnionArray8_32.regular_index(tags)).dtype == np.dtype(
        np.int32
    )
    assert np.asarray(ak.layout.UnionArray8_U32.regular_index(tags)).tolist() == [
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    assert np.asarray(ak.layout.UnionArray8_U32.regular_index(tags)).dtype == np.dtype(
        np.uint32
    )
    assert np.asarray(ak.layout.UnionArray8_64.regular_index(tags)).tolist() == [
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    assert np.asarray(ak.layout.UnionArray8_64.regular_index(tags)).dtype == np.dtype(
        np.int64
    )

    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])
    assert np.asarray(array.tags).tolist() == [1, 1, 0, 0, 1, 0, 1, 1]
    assert np.asarray(array.tags).dtype == np.dtype(np.int8)
    assert np.asarray(array.index).tolist() == [0, 1, 0, 1, 2, 2, 4, 3]
    assert np.asarray(array.index).dtype == np.dtype(np.int32)
    assert type(array.contents) is list
    assert [ak.to_list(x) for x in array.contents] == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        ["one", "two", "three", "four", "five"],
    ]
    assert array.numcontents == 2
    assert ak.to_list(array.content(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(array.content(1)) == ["one", "two", "three", "four", "five"]
    assert ak.to_list(array.project(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(array.project(1)) == ["one", "two", "three", "five", "four"]
    repr(array)

    assert ak.to_list(array[0]) == "one"
    assert ak.to_list(array[1]) == "two"
    assert ak.to_list(array[2]) == [1.1, 2.2, 3.3]
    assert ak.to_list(array[3]) == []
    assert ak.to_list(array[4]) == "three"
    assert ak.to_list(array[5]) == [4.4, 5.5]
    assert ak.to_list(array[6]) == "five"
    assert ak.to_list(array[7]) == "four"

    assert ak.to_list(array) == [
        "one",
        "two",
        [1.1, 2.2, 3.3],
        [],
        "three",
        [4.4, 5.5],
        "five",
        "four",
    ]
    assert ak.to_list(array[1:-1]) == [
        "two",
        [1.1, 2.2, 3.3],
        [],
        "three",
        [4.4, 5.5],
        "five",
    ]
    assert ak.to_list(array[2:-2]) == [[1.1, 2.2, 3.3], [], "three", [4.4, 5.5]]
    assert ak.to_list(array[::2]) == ["one", [1.1, 2.2, 3.3], "three", "five"]
    assert ak.to_list(array[::2, 1:]) == ["ne", [2.2, 3.3], "hree", "ive"]
    assert ak.to_list(array[:, :-1]) == [
        "on",
        "tw",
        [1.1, 2.2],
        [],
        "thre",
        [4.4],
        "fiv",
        "fou",
    ]

    content2 = ak.from_iter(
        [{"x": 0, "y": []}, {"x": 1, "y": [1.1]}, {"x": 2, "y": [1.1, 2.2]}],
        highlevel=False,
    )
    content3 = ak.from_iter(
        [
            {"x": 0.0, "y": "zero", "z": False},
            {"x": 1.1, "y": "one", "z": True},
            {"x": 2.2, "y": "two", "z": False},
            {"x": 3.3, "y": "three", "z": True},
            {"x": 4.4, "y": "four", "z": False},
        ],
        highlevel=False,
    )
    array2 = ak.layout.UnionArray8_32(tags, index, [content2, content3])
    assert ak.to_list(array2) == [
        {"x": 0.0, "y": "zero", "z": False},
        {"x": 1.1, "y": "one", "z": True},
        {"x": 0, "y": []},
        {"x": 1, "y": [1.1]},
        {"x": 2.2, "y": "two", "z": False},
        {"x": 2, "y": [1.1, 2.2]},
        {"x": 4.4, "y": "four", "z": False},
        {"x": 3.3, "y": "three", "z": True},
    ]
    assert ak.to_list(array2["x"]) == [0.0, 1.1, 0, 1, 2.2, 2, 4.4, 3.3]
    assert ak.to_list(array2["y"]) == [
        "zero",
        "one",
        [],
        [1.1],
        "two",
        [1.1, 2.2],
        "four",
        "three",
    ]
    assert ak.to_list(array2[:, "y", 1:]) == [
        "ero",
        "ne",
        [],
        [],
        "wo",
        [2.2],
        "our",
        "hree",
    ]
    assert ak.to_list(array2["y", :, 1:]) == [
        "ero",
        "ne",
        [],
        [],
        "wo",
        [2.2],
        "our",
        "hree",
    ]
    with pytest.raises(ValueError) as err:
        array2[:, 1:, "y"]
    assert str(err.value).startswith("in NumpyArray, too many dimensions in slice")
    with pytest.raises(ValueError) as err:
        array2["z"]
    assert str(err.value).startswith('key "z" does not exist (not in record)')

    array3 = ak.layout.UnionArray8_32(tags, index, [content3, content2])
    array4 = ak.layout.UnionArray8_32(
        tags, index, [content0, content1, content2, content3]
    )
    assert set(content2.keys()) == set(["x", "y"])
    assert set(content3.keys()) == set(["x", "y", "z"])
    assert set(array2.keys()) == set(["x", "y"])
    assert set(array3.keys()) == set(["x", "y"])
    assert array4.keys() == []


def test_identities():
    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])

    array.setidentities()
    assert np.asarray(array.identities).tolist() == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
    ]
    assert np.asarray(array.content(0).identities).tolist() == [[2], [3], [5]]
    assert np.asarray(array.content(1).identities).tolist() == [[0], [1], [4], [7], [6]]


def test_fromiter():
    builder = ak.layout.ArrayBuilder()

    builder.integer(0)
    builder.integer(1)
    builder.integer(2)
    builder.beginlist()
    builder.endlist()
    builder.beginlist()
    builder.real(1.1)
    builder.endlist()
    builder.beginlist()
    builder.real(1.1)
    builder.real(2.2)
    builder.endlist()
    builder.beginlist()
    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)
    builder.endlist()

    assert ak.to_list(builder.snapshot()) == [
        0,
        1,
        2,
        [],
        [1.1],
        [1.1, 2.2],
        [1.1, 2.2, 3.3],
    ]

    assert ak.to_list(
        ak.from_iter([0, 1, 2, [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3]])
    ) == [0, 1, 2, [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3]]
    assert ak.to_list(
        ak.from_iter(
            [
                0,
                1,
                2,
                [],
                "zero",
                [1.1],
                "one",
                [1.1, 2.2],
                "two",
                [1.1, 2.2, 3.3],
                "three",
            ]
        )
    ) == [
        0,
        1,
        2,
        [],
        "zero",
        [1.1],
        "one",
        [1.1, 2.2],
        "two",
        [1.1, 2.2, 3.3],
        "three",
    ]
    assert ak.to_list(
        ak.from_json(
            '[0, 1, 2, [], "zero", [1.1], "one", [1.1, 2.2], "two", [1.1, 2.2, 3.3], "three"]'
        )
    ) == [
        0,
        1,
        2,
        [],
        "zero",
        [1.1],
        "one",
        [1.1, 2.2],
        "two",
        [1.1, 2.2, 3.3],
        "three",
    ]
    assert (
        ak.to_json(
            ak.from_json(
                '[0,1,2,[],"zero",[1.1],"one",[1.1,2.2],"two",[1.1,2.2,3.3],"three"]'
            )
        )
        == '[0,1,2,[],"zero",[1.1],"one",[1.1,2.2],"two",[1.1,2.2,3.3],"three"]'
    )
