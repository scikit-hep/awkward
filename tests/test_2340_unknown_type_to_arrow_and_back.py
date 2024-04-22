# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# from __future__ import annotations
from __future__ import annotations

import pytest

pyarrow = pytest.importorskip("pyarrow")

import awkward as ak
from awkward.types import ListType, OptionType, UnknownType


def test_bare_unknown():
    unk_array = ak.Array([[], []])  # sub-arrays are type unknown
    assert unk_array[0].type.content == UnknownType()
    unk_array_arrow = ak.to_arrow(unk_array)
    assert len(unk_array_arrow) == 2
    # This array has one field, where we can git at pyarrow storage info
    field = unk_array_arrow.type.storage_type.field(0)
    assert field.type.storage_type == pyarrow.null()
    assert field.nullable  # Must be nullable to be valid in Arrow
    orig_array = ak.from_arrow(unk_array_arrow)
    assert len(orig_array) == 2
    assert orig_array.type == unk_array.type


def test_option_unknown():
    unk_option_array = ak.Array([[None, None], []])  # type is ?unknown
    assert unk_option_array.type.content == ListType(OptionType(UnknownType()))
    unk_opt_array_arrow = ak.to_arrow(unk_option_array)
    field = unk_opt_array_arrow.type.storage_type.field(0)
    assert field.type.storage_type == pyarrow.null()
    assert field.nullable  # Must be nullable to be valid in Arrow
    orig_array = ak.from_arrow(unk_opt_array_arrow)
    assert orig_array.type == unk_option_array.type
