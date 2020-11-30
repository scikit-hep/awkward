# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import os
import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_fromstring():
    a = ak.from_json("[[1.1, 2.2, 3], [], [4, 5.5]]")
    assert ak.to_list(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        ak.from_json("[[1.1, 2.2, 3], [blah], [4, 5.5]]")


def test_fromfile(tmp_path):
    with open(os.path.join(str(tmp_path), "tmp1.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], [], [4, 5.5]]")

    a = ak.from_json(os.path.join(str(tmp_path), "tmp1.json"))
    assert ak.to_list(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        ak.from_json("nonexistent.json")

    with open(os.path.join(str(tmp_path), "tmp2.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], []], [4, 5.5]]")

    with pytest.raises(ValueError):
        ak.from_json(os.path.join(str(tmp_path), "tmp2.json"))


def test_tostring():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(-1, 7))
    offsetsA = np.arange(0, 2 * 3 * 5 + 5, 5)
    offsetsB = np.arange(0, 2 * 3 + 3, 3)
    startsA, stopsA = offsetsA[:-1], offsetsA[1:]
    startsB, stopsB = offsetsB[:-1], offsetsB[1:]

    listoffsetarrayA32 = ak.layout.ListOffsetArray32(
        ak.layout.Index32(offsetsA), content
    )
    listarrayA32 = ak.layout.ListArray32(
        ak.layout.Index32(startsA), ak.layout.Index32(stopsA), content
    )
    modelA = np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7)

    listoffsetarrayB32 = ak.layout.ListOffsetArray32(
        ak.layout.Index32(offsetsB), listoffsetarrayA32
    )
    listarrayB32 = ak.layout.ListArray32(
        ak.layout.Index32(startsB), ak.layout.Index32(stopsB), listarrayA32
    )
    modelB = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    assert content.tojson() == json.dumps(ak.to_list(content), separators=(",", ":"))
    assert listoffsetarrayA32.tojson() == json.dumps(
        modelA.tolist(), separators=(",", ":")
    )
    assert listoffsetarrayB32.tojson() == json.dumps(
        modelB.tolist(), separators=(",", ":")
    )
    ak.to_json(ak.from_json("[[1.1,2.2,3],[],[4,5.5]]")) == "[[1.1,2.2,3],[],[4,5.5]]"


def test_tofile(tmp_path):
    ak.to_json(
        ak.from_json("[[1.1,2.2,3],[],[4,5.5]]"),
        os.path.join(str(tmp_path), "tmp1.json"),
    )

    with open(os.path.join(str(tmp_path), "tmp1.json"), "r") as f:
        f.read() == "[[1.1,2.2,3],[],[4,5.5]]"


def test_fromiter():
    assert ak.to_list(ak.from_iter([True, True, False, False, True])) == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert ak.to_list(ak.from_iter([5, 4, 3, 2, 1])) == [5, 4, 3, 2, 1]
    assert ak.to_list(ak.from_iter([5, 4, 3.14, 2.22, 1.23])) == [
        5.0,
        4.0,
        3.14,
        2.22,
        1.23,
    ]
    assert ak.to_list(ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert ak.to_list(
        ak.from_iter(
            [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
        )
    ) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]


def test_numpy():
    a = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert ak.to_list(a) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        ak.to_json(a)
        == "[[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]],[[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]]"
    )

    b = ak.layout.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
                [[10.1, 20.2, 30.3], [40.4, 50.5, 60.6]],
            ]
        )
    )
    assert (
        ak.to_json(b)
        == "[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,30.3],[40.4,50.5,60.6]]]"
    )

    c = ak.layout.NumpyArray(np.array([[True, False, True], [False, False, True]]))
    assert ak.to_json(c) == "[[true,false,true],[false,false,true]]"
