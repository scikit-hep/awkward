# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# from __future__ import annotations
from __future__ import annotations

import io

import pytest

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import awkward as ak
from awkward.operations import to_list
from awkward.types import ListType, OptionType, UnknownType


def test_bare_unknown():
    unk_array = ak.Array([[], []])  # sub-arrays are type unknown
    assert unk_array[0].type.content == UnknownType()

    unk_array_arrow = ak.to_arrow(unk_array)
    # This array has one field, where we can git at pyarrow storage info
    field = unk_array_arrow.type.storage_type.field(0)
    assert field.type.storage_type == pyarrow.null()
    assert field.nullable  # Must be nullable to be valid in Arrow
    array_is_valid_within_parquet(unk_array_arrow)

    orig_array = ak.from_arrow(unk_array_arrow)
    assert to_list(orig_array) == [[], []]
    assert orig_array.type == unk_array.type


def test_option_unknown():
    unk_option_array = ak.Array([[None, None], []])  # type is ?unknown
    assert unk_option_array.type.content == ListType(OptionType(UnknownType()))

    unk_opt_array_arrow = ak.to_arrow(unk_option_array)
    field = unk_opt_array_arrow.type.storage_type.field(0)
    assert field.type.storage_type == pyarrow.null()
    assert field.nullable  # Must be nullable to be valid in Arrow
    array_is_valid_within_parquet(unk_opt_array_arrow)

    orig_array = ak.from_arrow(unk_opt_array_arrow)
    assert orig_array.type == unk_option_array.type
    assert to_list(orig_array) == [[None, None], []]


def test_toplevel_unknown():
    unk_array = ak.Array([])  # type is unknown
    assert unk_array.type.content == UnknownType()

    unk_array_arrow = ak.to_arrow(unk_array)
    assert len(unk_array_arrow) == 0
    # Note: we cannot test the nullability of this arrow array, since it has zero fields
    # field = unk_array_arrow.type.storage_type.field(0)
    assert unk_array_arrow.type.num_fields == 0  # (just demonstrating the above)
    # But we can still give Parquet conversion a shot
    array_is_valid_within_parquet(unk_array_arrow)

    orig_array = ak.from_arrow(unk_array_arrow)
    assert orig_array.type == unk_array.type
    assert to_list(orig_array) == []


def array_is_valid_within_parquet(arrow_array):
    """
    Helper function that writes the given array to a Parquet table.
    Prior to 2340, this would raise:
    pyarrow.lib.ArrowInvalid: NullType Arrow field must be nullable
    """
    table = pyarrow.Table.from_arrays([arrow_array], names=["col1"])
    table.validate(
        full=True
    )  # Frustratingly, this does *not* produce the anticipated exception, but..
    temp = io.BytesIO()
    pq.write_table(table, temp)  # *this does* perform the validation we need.
    assert len(temp.getbuffer()) > 0
