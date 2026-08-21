# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


@pytest.mark.parametrize(
    "strings",
    [
        ["hello", "world"],
        ["", "abc"],
        ["café", "naïve"],  # two-byte UTF-8
        ["κόσμος", "Привет"],  # two-byte UTF-8
        ["日本語", "中文"],  # three-byte UTF-8
        ["🙂", "🚀"],  # four-byte UTF-8
        ["a🙂é日", ""],  # mixed widths
    ],
)
def test_numba_arraybuilder_append_string(strings):
    @numba.njit
    def build(values, builder):
        for value in values:
            builder.append(value)
        return builder

    typed_strings = numba.typed.List(strings)
    result = build(typed_strings, ak.ArrayBuilder()).snapshot()

    assert result.to_list() == strings


def test_numba_arraybuilder_string():
    @numba.njit
    def build(values, builder):
        for value in values:
            builder.string(value)
        return builder

    strings = ["hello", "café", "日本語", "🙂"]
    values = numba.typed.List(strings)

    result = build(values, ak.ArrayBuilder()).snapshot()

    assert result.to_list() == strings
