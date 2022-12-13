# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_0166_IndexedOptionArray():
    array = ak.highlevel.Array(
        [[2, 3, 5], None, [], [7, 11], None, [13], None, [17, 19]]
    )
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        30,
        None,
        1,
        77,
        None,
        13,
        None,
        323,
    ]

    array = ak.highlevel.Array(
        [[[2, 3], [5]], None, [], [[7], [11]], None, [[13]], None, [[17, 19]]]
    )
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        [6, 5],
        None,
        [],
        [7, 11],
        None,
        [13],
        None,
        [323],
    ]

    array = ak.highlevel.Array(
        [[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]]
    )
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        [6, None, 5],
        [],
        [7, 11],
        [13],
        [None, 17, 19],
    ]

    array = ak.highlevel.Array([[6, None, 5], [], [7, 11], [13], [None, 17, 19]])
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        30,
        1,
        77,
        13,
        323,
    ]


def test_0166_ByteMaskedArray():
    content = ak.operations.from_iter(
        [[2, 3, 5], [999], [], [7, 11], [], [13], [123, 999], [17, 19]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8))
    array = ak.highlevel.Array(
        ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    )
    assert to_list(array) == [
        [2, 3, 5],
        None,
        [],
        [7, 11],
        None,
        [13],
        None,
        [17, 19],
    ]
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        30,
        None,
        1,
        77,
        None,
        13,
        None,
        323,
    ]

    content = ak.operations.from_iter(
        [
            [[2, 3], [5]],
            [[999]],
            [],
            [[7], [11]],
            [],
            [[13]],
            [[123], [999]],
            [[17, 19]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8))
    array = ak.highlevel.Array(
        ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    )
    assert to_list(array) == [
        [[2, 3], [5]],
        None,
        [],
        [[7], [11]],
        None,
        [[13]],
        None,
        [[17, 19]],
    ]
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        [6, 5],
        None,
        [],
        [7, 11],
        None,
        [13],
        None,
        [323],
    ]

    content = ak.operations.from_iter(
        [[2, 3], [999], [5], [7], [11], [13], [], [17], [19]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8))
    bytemasked = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    array = ak.highlevel.Array(ak.contents.ListOffsetArray(offsets, bytemasked))
    array = ak.highlevel.Array(
        [[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]]
    )
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        [6, None, 5],
        [],
        [7, 11],
        [13],
        [None, 17, 19],
    ]

    content = ak.operations.from_iter(
        [6, None, 5, 7, 11, 13, None, 17, 19], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8))
    bytemasked = ak.contents.ByteMaskedArray.simplified(mask, content, valid_when=False)
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    array = ak.highlevel.Array(ak.contents.ListOffsetArray(offsets, bytemasked))
    assert to_list(array) == [[6, None, 5], [], [7, 11], [13], [None, 17, 19]]
    assert to_list(ak.operations.prod(array, axis=-1)) == [
        30,
        1,
        77,
        13,
        323,
    ]


def test_0167_strings():
    array = ak.highlevel.Array(["one", "two", "three", "two", "two", "one", "three"])
    assert to_list(array == "two") == [False, True, False, True, True, False, False]
    assert to_list("two" == array) == [False, True, False, True, True, False, False]
    assert to_list(array == ["two"]) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert to_list(["two"] == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert to_list(array == ak.highlevel.Array(["two"])) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert to_list(ak.highlevel.Array(["two"]) == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]

    array = ak.highlevel.Array(
        [["one", "two", "three"], [], ["two"], ["two", "one"], ["three"]]
    )
    assert to_list(array == "two") == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list("two" == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(array == ["two"]) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(["two"] == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(array == ak.highlevel.Array(["two"])) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(ak.highlevel.Array(["two"]) == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]

    array = ak.highlevel.Array(
        [["one", "two", "three"], [], ["two"], ["two", "one"], ["three"]]
    )
    assert to_list(array == ["three", "two", "one", "one", "three"]) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert to_list(["three", "two", "one", "one", "three"] == array) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert to_list(
        array == ak.highlevel.Array(["three", "two", "one", "one", "three"])
    ) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert to_list(
        ak.highlevel.Array(["three", "two", "one", "one", "three"]) == array
    ) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]


def test_0167_bytestrings():
    array = ak.highlevel.Array(
        [b"one", b"two", b"three", b"two", b"two", b"one", b"three"]
    )
    assert to_list(array == b"two") == [False, True, False, True, True, False, False]
    assert to_list(b"two" == array) == [False, True, False, True, True, False, False]
    assert to_list(array == [b"two"]) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert to_list([b"two"] == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert to_list(array == ak.highlevel.Array([b"two"])) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert to_list(ak.highlevel.Array([b"two"]) == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]

    array = ak.highlevel.Array(
        [[b"one", b"two", b"three"], [], [b"two"], [b"two", b"one"], [b"three"]]
    )
    assert to_list(array == b"two") == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(b"two" == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(array == [b"two"]) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list([b"two"] == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(array == ak.highlevel.Array([b"two"])) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert to_list(ak.highlevel.Array([b"two"]) == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]

    array = ak.highlevel.Array(
        [[b"one", b"two", b"three"], [], [b"two"], [b"two", b"one"], [b"three"]]
    )
    assert to_list(array == [b"three", b"two", b"one", b"one", b"three"]) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert to_list([b"three", b"two", b"one", b"one", b"three"] == array) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert to_list(
        array == ak.highlevel.Array([b"three", b"two", b"one", b"one", b"three"])
    ) == [[False, False, True], [], [False], [False, True], [True]]
    assert to_list(
        ak.highlevel.Array([b"three", b"two", b"one", b"one", b"three"]) == array
    ) == [[False, False, True], [], [False], [False, True], [True]]
