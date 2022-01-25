# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import os
import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_fromstring():
    array = ak._v2.operations.convert.from_json("[[1.1, 2.2, 3], [], [4, 5.5]]")
    assert array.tolist() == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        ak._v2.operations.convert.from_json("[[1.1, 2.2, 3], [blah], [4, 5.5]]")


def test_fromfile(tmp_path):
    with open(os.path.join(str(tmp_path), "tmp1.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], [], [4, 5.5]]")

    array = ak._v2.operations.io.from_json_file(
        os.path.join(str(tmp_path), "tmp1.json")
    )
    assert array.tolist() == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(IOError):
        ak._v2.operations.io.from_json_file("nonexistent.json")

    with open(os.path.join(str(tmp_path), "tmp2.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], []], [4, 5.5]]")

    with pytest.raises(ValueError):
        ak._v2.operations.io.from_json_file(os.path.join(str(tmp_path), "tmp2.json"))


@pytest.mark.skip(
    reason="AttributeError: 'NumpyArray' object has no attribute 'tojson'"
)
def test_tostring():
    content = ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(-1, 7))
    offsetsA = np.arange(0, 2 * 3 * 5 + 5, 5)
    offsetsB = np.arange(0, 2 * 3 + 3, 3)

    listoffsetarrayA32 = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index32(offsetsA), content
    )
    modelA = np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7)

    listoffsetarrayB32 = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index32(offsetsB), listoffsetarrayA32
    )
    modelB = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    assert content.tojson() == json.dumps(ak.to_list(content), separators=(",", ":"))
    assert listoffsetarrayA32.tojson() == json.dumps(
        modelA.tolist(), separators=(",", ":")
    )
    assert listoffsetarrayB32.tojson() == json.dumps(
        modelB.tolist(), separators=(",", ":")
    )
    assert (
        ak._v2.operations.convert.to_json(
            ak._v2.operations.convert.from_json("[[1.1,2.2,3],[],[4,5.5]]")
        )
        == "[[1.1,2.2,3.0],[],[4.0,5.5]]"
    )


@pytest.mark.skip(
    reason="awkward/_v2/operations/convert/ak_to_json.py:21: NotImplementedError"
)
def test_tofile(tmp_path):
    ak._v2.operations.convert.to_json(
        ak._v2.operations.convert.from_json("[[1.1,2.2,3],[],[4,5.5]]"),
        os.path.join(str(tmp_path), "tmp1.json"),
    )

    with open(os.path.join(str(tmp_path), "tmp1.json")) as f:
        assert f.read() == "[[1.1,2.2,3.0],[],[4.0,5.5]]"


def test_fromiter():
    assert ak._v2.operations.convert.from_iter(
        [True, True, False, False, True]
    ).tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert ak._v2.operations.convert.from_iter([5, 4, 3, 2, 1]).tolist() == [
        5,
        4,
        3,
        2,
        1,
    ]
    assert ak._v2.operations.convert.from_iter([5, 4, 3.14, 2.22, 1.23]).tolist() == [
        5.0,
        4.0,
        3.14,
        2.22,
        1.23,
    ]
    assert ak._v2.operations.convert.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    ).tolist() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert ak._v2.operations.convert.from_iter(
        [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    ).tolist() == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]


@pytest.mark.skip(
    reason="awkward/_v2/operations/convert/ak_to_json.py:21: NotImplementedError"
)
def test_numpy():
    a = ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert a.tolist() == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        ak._v2.operations.convert.to_json(a)
        == "[[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]],[[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]]"
    )

    b = ak._v2.contents.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
                [[10.1, 20.2, 30.3], [40.4, 50.5, 60.6]],
            ]
        )
    )
    assert (
        ak._v2.operations.convert.to_json(b)
        == "[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,30.3],[40.4,50.5,60.6]]]"
    )

    c = ak._v2.contents.NumpyArray(
        np.array([[True, False, True], [False, False, True]])
    )
    assert (
        ak._v2.operations.convert.to_json(c) == "[[true,false,true],[false,false,true]]"
    )
