# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# from __future__ import annotations
from __future__ import annotations

import io

import numpy as np
import pytest

import awkward as ak
from awkward.contents import (
    EmptyArray,
    NumpyArray,
    RecordArray,
    RegularArray,
    UnionArray,
)
from awkward.operations import to_list
from awkward.types import ListType, OptionType, UnknownType

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_bare_unknown():
    unk_array = ak.Array([[], []])  # sub-arrays are type unknown
    assert unk_array[0].type.content == UnknownType()

    unk_array_arrow = ak.to_arrow(unk_array)
    # This array has one field, where we can get at pyarrow storage info
    if hasattr(unk_array_arrow.type.storage_type, "field"):
        # We support older versions of pyarrow that lack a field method
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
    if hasattr(unk_opt_array_arrow.type.storage_type, "field"):
        field = unk_opt_array_arrow.type.storage_type.field(0)
        assert field.type.storage_type == pyarrow.null()
        assert field.nullable  # Nullable but this time it's because we're ?unknown
    array_is_valid_within_parquet(unk_opt_array_arrow)

    orig_array = ak.from_arrow(unk_opt_array_arrow)
    assert orig_array.type == unk_option_array.type
    assert to_list(orig_array) == [[None, None], []]

    # This is different for ... reasons:
    ua1 = ak.Array([[], [], [], [None]])[0:3]
    assert ua1.type.content == ListType(OptionType(UnknownType()))
    ua1a = ak.to_arrow(ua1)
    if hasattr(ua1a.type.storage_type, "field"):
        field = ua1a.type.storage_type.field(0)
        assert field.type.storage_type == pyarrow.null()
        assert field.nullable  # Like above, still nullable
    ua1aa = ak.from_arrow(ua1a)
    assert ua1aa.type.content == ListType(OptionType(UnknownType()))
    assert len(ua1aa) == 3


def test_toplevel_unknown():
    unk_array = ak.Array([])
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

    ua2 = ak.Array([None])[0:0]  # This is a top-level Option<EmptyArray>
    assert ua2.type.content == OptionType(UnknownType())
    ua2a = ak.to_arrow(ua2)
    ua2aa = ak.from_arrow(ua2a)
    assert ua2aa.type.content == OptionType(UnknownType())


def test_recordarray_with_unknowns():
    a = RecordArray([EmptyArray(), NumpyArray([])], ["x", "y"], length=0)
    arw = ak.to_arrow(a)
    if hasattr(arw.type.storage_type, "field"):
        assert arw.type.storage_type.field(0).nullable
    array_is_valid_within_parquet(arw)
    # This is a strange, laboratory kind of object.
    # It seems unlikely to be found in the wild.
    # I'm not sure what other tests here would be meaningful.


def test_table_with_unknowns():
    a = RecordArray([EmptyArray(), NumpyArray([1, 2])], ["x", "y"])
    # Again this is a strange one!
    table = ak.to_arrow_table(a)
    assert table.field(0).nullable
    temp = io.BytesIO()
    pq.write_table(table, temp)


def test_regulararray_with_unknown():
    a = RegularArray(EmptyArray(), 0)
    # RegularArray is helpful.
    # But when it's given an EmptyArray to recycle, it becomes absolutely unhelpful.
    arw = ak.to_arrow(a)
    if hasattr(arw.type.storage_type, "field"):
        assert arw.type.storage_type.field(0).nullable
    assert to_list(arw) == []
    array_is_valid_within_parquet(arw)


def test_unionarray_with_unknown():
    # Although a UnionArray with an EmptyArray content type has no application,
    # we can still exercise a code path this way.
    a = UnionArray(
        tags=ak.index.Index8(np.array([1, 1, 1], dtype=np.int8)),
        # tags: none of the elements are, or could be, taken from the EmptyArray.
        index=ak.index.Index64(np.array([0, 1, 2], dtype=np.int64)),
        contents=[
            EmptyArray(),
            NumpyArray([10, 20, 30]),
        ],
    )
    assert to_list(a) == [10, 20, 30]
    arw = ak.to_arrow(a)
    if hasattr(arw.type.storage_type, "field"):
        assert arw.type.storage_type.field(0).nullable
        assert not arw.type.storage_type.field(1).nullable
    # array_is_valid_within_parquet(arw)  # This fails for unrelated reasons.
    art = ak.from_arrow(arw)
    # round-trip is okay but the UnionArray is lost. Separate issue?
    assert to_list(art) == [10, 20, 30]


#### Helper method(s)


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
