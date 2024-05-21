from __future__ import annotations

import pytest

import numpy as np
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import awkward as ak
from awkward._connect.pyarrow_table_conv import (
    convert_awkward_arrow_table_to_native,
    convert_native_arrow_table_to_awkward,
    collect_ak_arr_type_metadata,
    awkward_arrow_field_to_native,
    native_arrow_field_to_akarraytype,
    array_with_replacement_type,
    AWKWARD_INFO_KEY,
)

nested_ints = ak.Array([[[[[1, 2, 3], [], [4, 5]] * 5] * 3] * 2])

struct_array = ak.Array([
    dict(lbl='item 1', idx=11, ids=[1, 2, 3]),
    dict(lbl='item 2', idx=12, ids=[51, 52]),
    dict(lbl='item 3', idx=13, ids=[61, 62, 63, 64]),
], with_name='my_items')

with_nulls = ak.Array([[11, None], [23], None])

option_types = ak.Array([[1, 2], [[10, 11, None], None, [3, 4, 5], ["one", "two"]], None])

indexed = ak.contents.IndexedArray(
    ak.index.Index64(np.array([2, 1, 0], dtype=np.uint64)),
    ak.contents.RecordArray(
        [
            ak.contents.ByteMaskedArray(
                ak.index.Index8(np.array([False, True, False]).view(np.int8)),
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3])
                ),
                valid_when=False,
            ),
            ak.contents.UnmaskedArray(
                ak.contents.ListOffsetArray(
                    ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                    ),
                ),
            ),
        ],
        ["x", "y"],
    ),
    parameters={"__array__": "categorical", "foo": "thap"},
)


@pytest.mark.parametrize("akarray", [nested_ints, struct_array, with_nulls, option_types, indexed])
def test_array_conversions(akarray):
    arrow_natv = ak.to_arrow(akarray, categorical_as_dictionary=True, extensionarray=False)
    arrow_extn = ak.to_arrow(akarray, categorical_as_dictionary=True, extensionarray=True)
    extn_field = pa.field("test_field", arrow_extn.type)

    # Convert to native pyarrow types:
    metadata = collect_ak_arr_type_metadata(extn_field)
    conv_natv_field = awkward_arrow_field_to_native(extn_field)
    assert conv_natv_field.type == arrow_natv.type
    as_natv = array_with_replacement_type(arrow_extn, conv_natv_field.type)
    assert as_natv == arrow_natv

    # Complete a round-trip, back to AwkwardArrowArray
    conv_extn_field = native_arrow_field_to_akarraytype(conv_natv_field, metadata)
    assert conv_extn_field.type == arrow_extn.type
    as_extn = array_with_replacement_type(as_natv, conv_extn_field.type)
    assert as_extn == arrow_extn


def test_table_conversion():
    ak_tbl_like = ak.Array(dict(
        struct_array=struct_array,
        with_nulls=with_nulls,
        option_types=option_types,
        indexed=indexed,
    ))
    natv_tbl = ak.to_arrow_table(ak_tbl_like, categorical_as_dictionary=True, extensionarray=False)
    extn_tbl = ak.to_arrow_table(ak_tbl_like, categorical_as_dictionary=True, extensionarray=True)

    conv_natv_tbl = convert_awkward_arrow_table_to_native(extn_tbl)
    assert conv_natv_tbl.schema == natv_tbl.schema  # This comparison does not include metadata
    assert conv_natv_tbl == natv_tbl
    assert AWKWARD_INFO_KEY in conv_natv_tbl.schema.metadata
    assert len(conv_natv_tbl["struct_array"].chunks[0].field(2).buffers()) == 4
    # This verifies that our conversions are zero-copy for the table data
    assert (extn_tbl["struct_array"].chunks[0].storage.field(2).buffers()[3].address
            == conv_natv_tbl["struct_array"].chunks[0].field(2).buffers()[3].address)

    conv_extn_tbl = convert_native_arrow_table_to_awkward(conv_natv_tbl)
    assert conv_extn_tbl.schema == extn_tbl.schema
    assert conv_extn_tbl == extn_tbl  # Full Round-trip verification
    assert AWKWARD_INFO_KEY not in conv_extn_tbl.schema.metadata
    assert (extn_tbl["struct_array"].chunks[0].storage.field(2).buffers()[3].address
            == conv_extn_tbl["struct_array"].chunks[0].storage.field(2).buffers()[3].address)
