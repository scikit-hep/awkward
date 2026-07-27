# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

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
