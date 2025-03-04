# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import io
import os

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import awkward as ak
from awkward._connect.pyarrow.table_conv import (
    AWKWARD_INFO_KEY,
    array_with_replacement_type,
    awkward_arrow_field_to_native,
    collect_ak_arr_type_metadata,
    convert_awkward_arrow_table_to_native,
    convert_native_arrow_table_to_awkward,
    native_arrow_field_to_akarraytype,
)
from awkward.operations import to_list

nested_ints = ak.Array([[[[[1, 2, 3], [], [4, 5]] * 5] * 3] * 2])

struct_array = ak.Array(
    [
        {"lbl": "item 1", "idx": 11, "ids": [1, 2, 3]},
        {"lbl": "item 2", "idx": 12, "ids": [51, 52]},
        {"lbl": "item 3", "idx": 13, "ids": [61, 62, 63, 64]},
    ],
    with_name="my_items",
)

with_nulls = ak.Array([[11, None], [23], None])

option_types = ak.Array(
    [[1, 2], [[10, 11, None], None, [3, 4, 5], ["one", "two"]], None]
)

(indexed, indexed_tuple) = (
    # Unique for this test, contains nested types within the
    ak.contents.IndexedArray(
        ak.index.Index64(np.array([2, 1, 0], dtype=np.uint64)),
        ak.contents.RecordArray(
            [
                ak.contents.ByteMaskedArray(
                    ak.index.Index8(np.array([False, True, False]).view(np.int8)),
                    ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3])),
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
            None if is_tuple else ["x", "y"],
        ),
        parameters={"__array__": "categorical", "foo": "thap"},
    )
    for is_tuple in (False, True)
)

(records, records_tuple) = (
    ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(
                np.array([1.1, 2.2, 3.3]), parameters={"foo": "inner1"}
            ),
            ak.contents.ListOffsetArray(
                ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"foo": "inner2"}
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"foo": "outer"},
    )
    for is_tuple in (False, True)
)


@pytest.mark.parametrize(
    "akarray, as_dict",
    [
        (nested_ints, False),
        (struct_array, False),
        (with_nulls, False),
        (option_types, False),
        (indexed, True),
        (indexed_tuple, True),
        (records, True),
        (records_tuple, True),
        (indexed, False),
        (indexed_tuple, False),
        (records, False),
        (records_tuple, False),
    ],
)
def test_array_conversions(akarray, as_dict):
    arrow_natv = ak.to_arrow(
        akarray, categorical_as_dictionary=as_dict, extensionarray=False
    )
    arrow_extn = ak.to_arrow(
        akarray, categorical_as_dictionary=as_dict, extensionarray=True
    )
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
    assert (
        conv_extn_field.type._metadata_as_dict() == arrow_extn.type._metadata_as_dict()
    )
    conv_metadata = collect_ak_arr_type_metadata(conv_extn_field)
    assert conv_metadata == metadata
    # assert_equal_arrowextntypes(conv_extn_field.type, arrow_extn.type)
    as_extn = array_with_replacement_type(as_natv, conv_extn_field.type)
    assert as_extn == arrow_extn
    assert as_extn.type._metadata_as_dict() == arrow_extn.type._metadata_as_dict()

    # And back to Awkward array
    rt_array = ak.from_arrow(as_extn, highlevel=True)
    assert to_list(rt_array) == to_list(akarray)

    # Deeper test of types
    akarray_high = ak.Array(akarray)
    if akarray_high.type.content.parameters.get("__categorical__", False) == as_dict:
        # as_dict is supposed to go hand-in-hand with __categorical__: True, and if it
        #  does not, we do not round-trip perfectly. So only test when this is set correctly.
        assert rt_array.type == akarray_high.type

        ak_type_str_orig = io.StringIO()
        ak_type_str_rtrp = io.StringIO()
        akarray_high.type.show(stream=ak_type_str_orig)
        rt_array.type.show(stream=ak_type_str_rtrp)
        if ak_type_str_orig.getvalue() != ak_type_str_rtrp.getvalue():
            print("  Original type:", ak_type_str_orig.getvalue())
            print("  Rnd-trip type:", ak_type_str_rtrp.getvalue())
        assert ak_type_str_orig.getvalue() == ak_type_str_rtrp.getvalue()


def test_table_conversion():
    ak_tbl_like = ak.Array(
        {
            "struct_array": struct_array,
            "with_nulls": with_nulls,
            "option_types": option_types,
            "indexed": indexed,
        }
    )
    natv_tbl = ak.to_arrow_table(
        ak_tbl_like, categorical_as_dictionary=True, extensionarray=False
    )
    extn_tbl = ak.to_arrow_table(
        ak_tbl_like, categorical_as_dictionary=True, extensionarray=True
    )

    conv_natv_tbl = convert_awkward_arrow_table_to_native(extn_tbl)
    assert (
        conv_natv_tbl.schema == natv_tbl.schema
    )  # This comparison does not include metadata
    assert conv_natv_tbl == natv_tbl
    assert AWKWARD_INFO_KEY in conv_natv_tbl.schema.metadata
    assert len(conv_natv_tbl["struct_array"].chunks[0].field(2).buffers()) == 4
    # This verifies that our conversions are zero-copy for the table data
    assert (
        extn_tbl["struct_array"].chunks[0].storage.field(2).buffers()[3].address
        == conv_natv_tbl["struct_array"].chunks[0].field(2).buffers()[3].address
    )

    conv_extn_tbl = convert_native_arrow_table_to_awkward(conv_natv_tbl)
    assert conv_extn_tbl.schema == extn_tbl.schema
    assert conv_extn_tbl == extn_tbl  # Full Round-trip verification
    assert AWKWARD_INFO_KEY not in conv_extn_tbl.schema.metadata
    assert (
        extn_tbl["struct_array"].chunks[0].storage.field(2).buffers()[3].address
        == conv_extn_tbl["struct_array"].chunks[0].storage.field(2).buffers()[3].address
    )


def test_selective_parquet(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    ak_tbl = ak.Array(
        {
            "with_nulls": with_nulls,
            "struct_array": struct_array,
            "indexed": indexed,
        }
    )
    ak.to_parquet(ak_tbl, filename)
    tbl_tr = ak.from_parquet(filename, columns=["struct_array", "indexed"])
    assert to_list(tbl_tr["struct_array"]) == to_list(ak_tbl["struct_array"])


@pytest.mark.parametrize("doit", [False, True])
def test_empty(tmp_path, doit):
    filename = os.path.join(tmp_path, "whatever.parquet")

    ak.to_parquet(ak.Array([{"x": 1, "y": 1.1}])[0:0], filename, extensionarray=doit)

    assert str(ak.from_parquet(filename).type) == "0 * {x: int64, y: float64}"
