# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import os
import pathlib

# FIXME: for float32
import re

import numpy as np
import pytest

import awkward as ak

simpledec = re.compile(r"\d*\.\d+")


def mround(match):
    return f"{float(match.group()):.1f}"


# re.sub(simpledec, mround, text)


def test_fromstring():
    array = ak.operations.from_json("[[1.1, 2.2, 3], [], [4, 5.5]]")
    assert array.to_list() == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        ak.operations.from_json("[[1.1, 2.2, 3], [blah], [4, 5.5]]")


def test_fromfile(tmp_path):
    with open(os.path.join(str(tmp_path), "tmp1.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], [], [4, 5.5]]")

    array = ak.operations.from_json(tmp_path / "tmp1.json")
    assert array.to_list() == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(IOError):
        ak.operations.from_json(pathlib.Path("nonexistent.json"))

    with open(os.path.join(str(tmp_path), "tmp2.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], []], [4, 5.5]]")

    with pytest.raises(ValueError):
        ak.operations.from_json(tmp_path / "tmp2.json")


def test_tostring():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(-1, 7))
    offsetsA = np.arange(0, 2 * 3 * 5 + 5, 5)
    offsetsB = np.arange(0, 2 * 3 + 3, 3)

    listoffsetarrayA32 = ak.contents.ListOffsetArray(
        ak.index.Index32(offsetsA), content
    )
    modelA = np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7)

    listoffsetarrayB32 = ak.contents.ListOffsetArray(
        ak.index.Index32(offsetsB), listoffsetarrayA32
    )
    modelB = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    assert ak.operations.to_json(content) == json.dumps(
        content.to_list(), separators=(",", ":")
    )
    assert ak.operations.to_json(listoffsetarrayA32) == json.dumps(
        modelA.tolist(), separators=(",", ":")
    )
    assert ak.operations.to_json(listoffsetarrayB32) == json.dumps(
        modelB.tolist(), separators=(",", ":")
    )
    assert (
        ak.operations.to_json(ak.operations.from_json("[[1.1,2.2,3],[],[4,5.5]]"))
        == "[[1.1,2.2,3.0],[],[4.0,5.5]]"
    )


def test_bytearray():
    array = ak.contents.NumpyArray(
        np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "byte"}
    )
    assert ak.operations.to_json(array, convert_bytes=bytes.decode) == '"hellothere"'


def test_complex():
    content = ak.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    assert (
        ak.operations.to_json(content, complex_record_fields=("r", "i"))
        == """[{"r":1.1,"i":0.1},{"r":2.2,"i":0.0},{"r":3.3,"i":0.0},{"r":4.4,"i":0.0},{"r":5.5,"i":0.0},{"r":6.6,"i":0.0},{"r":7.7,"i":0.0},{"r":8.8,"i":0.0},{"r":9.9,"i":0.0}]"""
    )

    array = ak.operations.from_json(
        '[{"r":1.1,"i":1.0},{"r":2.2,"i":2.0}]', complex_record_fields=("r", "i")
    )
    assert (
        ak.operations.to_json(array, complex_record_fields=("r", "i"))
        == """[{"r":1.1,"i":1.0},{"r":2.2,"i":2.0}]"""
    )

    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(-1, 7))
    assert (
        ak.operations.to_json(content, complex_record_fields=("r", "i"))
        == """[[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],[21,22,23,24,25,26,27],[28,29,30,31,32,33,34],[35,36,37,38,39,40,41],[42,43,44,45,46,47,48],[49,50,51,52,53,54,55],[56,57,58,59,60,61,62],[63,64,65,66,67,68,69],[70,71,72,73,74,75,76],[77,78,79,80,81,82,83],[84,85,86,87,88,89,90],[91,92,93,94,95,96,97],[98,99,100,101,102,103,104],[105,106,107,108,109,110,111],[112,113,114,115,116,117,118],[119,120,121,122,123,124,125],[126,127,128,129,130,131,132],[133,134,135,136,137,138,139],[140,141,142,143,144,145,146],[147,148,149,150,151,152,153],[154,155,156,157,158,159,160],[161,162,163,164,165,166,167],[168,169,170,171,172,173,174],[175,176,177,178,179,180,181],[182,183,184,185,186,187,188],[189,190,191,192,193,194,195],[196,197,198,199,200,201,202],[203,204,205,206,207,208,209]]"""
    )


def test_complex_with_nan_and_inf():
    content = ak.contents.NumpyArray(
        np.array(
            [
                (1.1 + 0.1j),
                2.2,
                3.3,
                (np.nan + 1j * np.nan),
                5.5,
                -np.inf,
                7.7,
                (np.inf + 1j * np.inf),
                9.9,
            ]
        )
    )
    # Note: NumPy '(np.inf + 1j*np.inf)' conversion is consistent with math:
    # >>> z = (np.inf+ 1j*np.inf)
    # >>> z
    # (nan+infj)
    # >>> z = (math.inf+1j*math.inf)
    # >>> z
    # (nan+infj)
    assert (
        ak.operations.to_json(
            content,
            complex_record_fields=("r", "i"),
            nan_string="Not a number",
            posinf_string="Inf",
            neginf_string="-Inf",
        )
        == """[{"r":1.1,"i":0.1},{"r":2.2,"i":0.0},{"r":3.3,"i":0.0},{"r":"Not a number","i":"Not a number"},{"r":5.5,"i":0.0},{"r":"-Inf","i":0.0},{"r":7.7,"i":0.0},{"r":"Not a number","i":"Inf"},{"r":9.9,"i":0.0}]"""
    )


def test_tofile(tmp_path):
    ak.operations.to_json(
        ak.operations.from_json("[[1.1,2.2,3],[],[4,5.5]]"),
        file=os.path.join(str(tmp_path), "tmp1.json"),
    )

    with open(os.path.join(str(tmp_path), "tmp1.json")) as f:
        assert f.read() == "[[1.1,2.2,3.0],[],[4.0,5.5]]"


def test_fromiter():
    assert ak.operations.from_iter([True, True, False, False, True]).to_list() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert ak.operations.from_iter([5, 4, 3, 2, 1]).to_list() == [
        5,
        4,
        3,
        2,
        1,
    ]
    assert ak.operations.from_iter([5, 4, 3.14, 2.22, 1.23]).to_list() == [
        5.0,
        4.0,
        3.14,
        2.22,
        1.23,
    ]
    assert ak.operations.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).to_list() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert ak.operations.from_iter(
        [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    ).to_list() == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]


def test_numpy():
    a = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert a.to_list() == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        ak.operations.to_json(a)
        == "[[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]],[[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]]"
    )

    b = ak.contents.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
                [[10.1, 20.2, 30.3], [40.4, 50.5, 60.6]],
            ]
        )
    )
    assert (
        ak.operations.to_json(b)
        == "[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,30.3],[40.4,50.5,60.6]]]"
    )
    b1 = ak.contents.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
                [[10.1, 20.2, 30.3], [40.4, 50.5, 60.6]],
            ],
            dtype=np.float32,
        )
    )
    assert (
        re.sub(simpledec, mround, ak.operations.to_json(b1))
        == "[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,30.3],[40.4,50.5,60.6]]]"
    )
    b2 = ak.contents.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, np.inf, 6.6]],
                [[10.1, 20.2, np.nan], [40.4, 50.5, -np.inf]],
            ]
        )
    )
    assert (
        ak.operations.to_json(
            b2,
            nan_string="Not a number",
            posinf_string="Inf",
            neginf_string="-Inf",
        )
        == """[[[1.1,2.2,3.3],[4.4,"Inf",6.6]],[[10.1,20.2,"Not a number"],[40.4,50.5,"-Inf"]]]"""
    )
    b3 = ak.contents.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
                [[10.1, 20.2, np.inf], [40.4, 50.5, 60.6]],
            ]
        )
    )
    assert (
        ak.operations.to_json(b3, posinf_string="Infinity")
        == '[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,"Infinity"],[40.4,50.5,60.6]]]'
    )
    b4 = ak.contents.NumpyArray(
        np.array(
            [
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
                [[10.1, 20.2, -np.inf], [40.4, 50.5, 60.6]],
            ]
        )
    )
    assert (
        ak.operations.to_json(b4, neginf_string="-Infinity")
        == '[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,"-Infinity"],[40.4,50.5,60.6]]]'
    )
    c = ak.contents.NumpyArray(np.array([[True, False, True], [False, False, True]]))
    assert ak.operations.to_json(c) == "[[true,false,true],[false,false,true]]"
