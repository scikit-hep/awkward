# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_numba_array_getitem_string_native():
    array = ak.Array(
        [
            "",
            "hello",
            "café",
            "κόσμος",
            "日本語",
            "🙂",
            "a🙂é日",
        ]
    )

    @numba.njit
    def getitem(array, index):
        value = array[index]
        return value, len(value), value == "🙂"

    for index, expected in enumerate(array.to_list()):
        value, length, is_smiley = getitem(array, index)

        assert value == expected
        assert length == len(expected)
        assert is_smiley == (expected == "🙂")


def test_numba_array_getitem_bytestring_native():
    values = [
        b"",
        b"hello",
        b"\x00\x01\x7f\x80\xff",
    ]
    array = ak.Array(values)

    @numba.njit
    def getitem(array, index):
        value = array[index]
        total = 0
        for byte in value:
            total += byte
        return value, len(value), total

    for index, expected in enumerate(values):
        value, length, total = getitem(array, index)

        assert value == expected
        assert length == len(expected)
        assert total == sum(expected)


def test_numba_array_getitem_string_operations():
    array = ak.Array(["hello", "café", "日本語", "🙂"])

    @numba.njit
    def f(array):
        return (
            array[0].upper(),
            array[1][1],
            array[2] + array[3],
            array[3] == "🙂",
        )

    assert f(array) == (
        "HELLO",
        "a",
        "日本語🙂",
        True,
    )


def array_from_raw_bytes(data, *, is_string):
    content_parameter = "char" if is_string else "byte"
    list_parameter = "string" if is_string else "bytestring"

    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.NumpyForm(
            "uint8",
            parameters={"__array__": content_parameter},
            form_key="node1",
        ),
        parameters={"__array__": list_parameter},
        form_key="node0",
    )

    return ak.from_buffers(
        form,
        1,
        {
            "node0-offsets": np.array([0, len(data)], dtype=np.int64),
            "node1-data": np.frombuffer(data, dtype=np.uint8),
        },
    )


@pytest.mark.parametrize(
    "data",
    [
        b"\xff",  # invalid leading byte
        b"\xc3",  # truncated two-byte sequence
        b"\xe2\x28\xa1",  # invalid continuation byte
        b"\xc0\xaf",  # overlong encoding
        b"\xed\xa0\x80",  # UTF-8 encoding of a surrogate
        b"\xf4\x90\x80\x80",  # code point above U+10FFFF
    ],
)
def test_numba_string_getitem_rejects_invalid_utf8(data):
    array = array_from_raw_bytes(data, is_string=True)

    @numba.njit
    def getitem(array):
        return array[0]

    with pytest.raises(ValueError):
        getitem(array)


@pytest.mark.parametrize(
    "data",
    [
        b"\xff",
        b"\xc3",
        b"\xe2\x28\xa1",
        b"\x00\xff",
    ],
)
def test_numba_bytestring_getitem_preserves_arbitrary_bytes(data):
    array = array_from_raw_bytes(data, is_string=False)

    @numba.njit
    def getitem(array):
        return array[0]

    assert getitem(array) == data
