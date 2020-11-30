# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_0166_IndexedOptionArray():
    array = ak.Array([[2, 3, 5], None, [], [7, 11], None, [13], None, [17, 19]])
    assert ak.to_list(ak.prod(array, axis=-1)) == [30, None, 1, 77, None, 13, None, 323]

    array = ak.Array(
        [[[2, 3], [5]], None, [], [[7], [11]], None, [[13]], None, [[17, 19]]]
    )
    assert ak.to_list(ak.prod(array, axis=-1)) == [
        [6, 5],
        None,
        [],
        [7, 11],
        None,
        [13],
        None,
        [323],
    ]

    array = ak.Array([[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]])
    ak.to_list(ak.prod(array, axis=-1)) == [
        [6, None, 5],
        [],
        [7, 11],
        [13],
        [None, 17, 19],
    ]

    array = ak.Array([[6, None, 5], [], [7, 11], [13], [None, 17, 19]])
    assert ak.to_list(ak.prod(array, axis=-1)) == [30, 1, 77, 13, 323]


def test_0166_ByteMaskedArray():
    content = ak.from_iter(
        [[2, 3, 5], [999], [], [7, 11], [], [13], [123, 999], [17, 19]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert ak.to_list(array) == [
        [2, 3, 5],
        None,
        [],
        [7, 11],
        None,
        [13],
        None,
        [17, 19],
    ]
    assert ak.to_list(ak.prod(array, axis=-1)) == [30, None, 1, 77, None, 13, None, 323]

    content = ak.from_iter(
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
    mask = ak.layout.Index8(np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert ak.to_list(array) == [
        [[2, 3], [5]],
        None,
        [],
        [[7], [11]],
        None,
        [[13]],
        None,
        [[17, 19]],
    ]
    assert ak.to_list(ak.prod(array, axis=-1)) == [
        [6, 5],
        None,
        [],
        [7, 11],
        None,
        [13],
        None,
        [323],
    ]

    content = ak.from_iter(
        [[2, 3], [999], [5], [7], [11], [13], [], [17], [19]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8))
    bytemasked = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    array = ak.Array(ak.layout.ListOffsetArray64(offsets, bytemasked))
    array = ak.Array([[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]])
    assert ak.to_list(ak.prod(array, axis=-1)) == [
        [6, None, 5],
        [],
        [7, 11],
        [13],
        [None, 17, 19],
    ]

    content = ak.from_iter([6, None, 5, 7, 11, 13, None, 17, 19], highlevel=False)
    mask = ak.layout.Index8(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8))
    bytemasked = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    array = ak.Array(ak.layout.ListOffsetArray64(offsets, bytemasked))
    assert ak.to_list(array) == [[6, None, 5], [], [7, 11], [13], [None, 17, 19]]
    assert ak.to_list(ak.prod(array, axis=-1)) == [30, 1, 77, 13, 323]


def test_0167_strings():
    array = ak.Array(["one", "two", "three", "two", "two", "one", "three"])
    assert ak.to_list(array == "two") == [False, True, False, True, True, False, False]
    assert ak.to_list("two" == array) == [False, True, False, True, True, False, False]
    assert ak.to_list(array == ["two"]) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert ak.to_list(["two"] == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert ak.to_list(array == ak.Array(["two"])) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert ak.to_list(ak.Array(["two"]) == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]

    array = ak.Array([["one", "two", "three"], [], ["two"], ["two", "one"], ["three"]])
    assert ak.to_list(array == "two") == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list("two" == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(array == ["two"]) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(["two"] == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(array == ak.Array(["two"])) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(ak.Array(["two"]) == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]

    array = ak.Array([["one", "two", "three"], [], ["two"], ["two", "one"], ["three"]])
    assert ak.to_list(array == ["three", "two", "one", "one", "three"]) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert ak.to_list(["three", "two", "one", "one", "three"] == array) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert ak.to_list(array == ak.Array(["three", "two", "one", "one", "three"])) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert ak.to_list(ak.Array(["three", "two", "one", "one", "three"]) == array) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]


def test_0167_bytestrings():
    array = ak.Array([b"one", b"two", b"three", b"two", b"two", b"one", b"three"])
    assert ak.to_list(array == b"two") == [False, True, False, True, True, False, False]
    assert ak.to_list(b"two" == array) == [False, True, False, True, True, False, False]
    assert ak.to_list(array == [b"two"]) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert ak.to_list([b"two"] == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert ak.to_list(array == ak.Array([b"two"])) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert ak.to_list(ak.Array([b"two"]) == array) == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]

    array = ak.Array(
        [[b"one", b"two", b"three"], [], [b"two"], [b"two", b"one"], [b"three"]]
    )
    assert ak.to_list(array == b"two") == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(b"two" == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(array == [b"two"]) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list([b"two"] == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(array == ak.Array([b"two"])) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]
    assert ak.to_list(ak.Array([b"two"]) == array) == [
        [False, True, False],
        [],
        [True],
        [True, False],
        [False],
    ]

    array = ak.Array(
        [[b"one", b"two", b"three"], [], [b"two"], [b"two", b"one"], [b"three"]]
    )
    assert ak.to_list(array == [b"three", b"two", b"one", b"one", b"three"]) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert ak.to_list([b"three", b"two", b"one", b"one", b"three"] == array) == [
        [False, False, True],
        [],
        [False],
        [False, True],
        [True],
    ]
    assert ak.to_list(
        array == ak.Array([b"three", b"two", b"one", b"one", b"three"])
    ) == [[False, False, True], [], [False], [False, True], [True]]
    assert ak.to_list(
        ak.Array([b"three", b"two", b"one", b"one", b"three"]) == array
    ) == [[False, False, True], [], [False], [False, True], [True]]
