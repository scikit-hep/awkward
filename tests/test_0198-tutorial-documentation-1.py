# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_singletons():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    assert ak.to_list(ak.singletons(array)) == [
        [1.1],
        [2.2],
        [],
        [3.3],
        [],
        [],
        [4.4],
        [5.5],
    ]

    assert ak.to_list(
        ak.singletons(ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]))
    ) == [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]
    assert ak.to_list(
        ak.singletons(ak.Array([[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]]))
    ) == [[[1.1], [2.2], []], [[3.3], []], [[]], [[4.4], [5.5]]]
    assert ak.to_list(
        ak.singletons(
            ak.Array([[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]])
        )
    ) == [[[[1.1], [2.2], []]], [[[3.3], []]], [[[]]], [[[4.4], [5.5]]]]


def test_firsts():
    assert ak.to_list(
        ak.firsts(
            ak.singletons(ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])), axis=1
        )
    ) == [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]
    assert (
        ak.to_list(
            ak.firsts(
                ak.singletons(
                    ak.Array([[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]])
                ),
                axis=2,
            )
        )
        == [[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]]
    )
    assert (
        ak.to_list(
            ak.firsts(
                ak.singletons(
                    ak.Array(
                        [[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]]
                    )
                ),
                axis=3,
            )
        )
        == [[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]]
    )


def test_allow_missing():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    ak.to_numpy(array)
    with pytest.raises(ValueError):
        ak.to_numpy(array, allow_missing=False)


def test_flatten0():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    assert ak.to_list(ak.flatten(array, axis=0)) == [1.1, 2.2, 3.3, 4.4, 5.5]

    content0 = ak.from_iter(
        [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5], highlevel=False
    )
    content1 = ak.from_iter(["one", None, "two", None, "three"], highlevel=False)
    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8(
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
            ),
            ak.layout.Index64(
                np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 4, 7], dtype=np.int64)
            ),
            [content0, content1],
        )
    )
    assert ak.to_list(array) == [
        1.1,
        "one",
        2.2,
        None,
        None,
        "two",
        3.3,
        None,
        None,
        None,
        4.4,
        "three",
        5.5,
    ]
    assert ak.to_list(ak.flatten(array, axis=0)) == [
        1.1,
        "one",
        2.2,
        "two",
        3.3,
        4.4,
        "three",
        5.5,
    ]
