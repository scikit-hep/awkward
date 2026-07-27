# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


@pytest.mark.parametrize(
    "array",
    [
        ak.Array([True, False, True]),
        ak.Array([1, 2, 3]),
        ak.Array([1.5, 2.5, 3.5]),
        ak.Array([[1, 2], [], [3]]),
        ak.Array(["hello", "world"]),
        ak.Array([b"hello", b"\x00\xff"]),
    ],
)
def test_numba_getitem_after_pointer_conversion(array):
    @numba.njit
    def getitem(array, index):
        return array[index]

    for index, expected in enumerate(array.to_list()):
        assert getitem(array, index) == expected


def test_numba_ragged_string_membership():
    array = ak.Array(
        [
            ["TRA", "TRB"],
            [],
            None,
            ["TRG", "TRA"],
        ]
    )

    @numba.njit
    def isin(array, haystack, builder):
        for row in array:
            builder.begin_list()

            if row is not None:
                for value in row:
                    builder.append(value in haystack)

            builder.end_list()

        return builder

    result = isin(
        array,
        ("TRA", "TRB"),
        ak.ArrayBuilder(),
    ).snapshot()

    assert result.to_list() == [
        [True, True],
        [],
        [],
        [False, True],
    ]
