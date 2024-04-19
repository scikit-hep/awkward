# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# from __future__ import annotations

import pytest
pyarrow = pytest.importorskip("pyarrow")

import awkward as ak


def test_bare_unknown():
    unk_array = ak.Array([])
    assert unk_array.type.content == ak.types.UnknownType()
    unk_array_arrow = ak.to_arrow(unk_array)
    assert len(unk_array_arrow) == 0
    assert unk_array_arrow.type.storage_type == pyarrow.null()
    orig_array = ak.from_arrow(unk_array_arrow)
    assert len(orig_array) == 0
    assert orig_array.type == unk_array.type


def test_option_unknown():
    unk_option_array = ak.Array([[None, None], []])[1]  # length 0, type is ?unknown
    assert unk_option_array.type.content == ak.types.OptionType(ak.types.UnknownType())
    unk_opt_array_arrow = ak.to_arrow(unk_option_array)
    assert len(unk_opt_array_arrow) == 0
    assert unk_opt_array_arrow.type.storage_type == pyarrow.null()
    orig_array = ak.from_arrow(unk_opt_array_arrow)
    assert len(orig_array) == 0
    assert orig_array.type == unk_option_array.type
