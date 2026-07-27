# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_numba_arraybuilder_append_string():
    @numba.njit
    def build(strings, builder):
        for value in strings:
            builder.append(value)
        return builder

    strings = numba.typed.List(["hello", "world"])
    result = build(strings, ak.ArrayBuilder()).snapshot()

    assert result.to_list() == ["hello", "world"]
