# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_arrow_table(
    array,
    explode_records=False,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
):
    pass


#     """
#     Args:
#         array: Data to convert to an Apache Arrow table.
#         explode_records (bool): If True, lists of records are written as
#             records of lists, so that nested fields become top-level fields
#             (which can be zipped when read back).
#         list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
#             if they're small enough, even if it means an extra conversion. Otherwise,
#             signed 32-bit #ak.layout.ListOffsetArray maps to Arrow `ListType` and
#             all others map to Arrow `LargeListType`.
#         string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
#         bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.

#     Converts an Awkward Array into an Apache Arrow table (`pyarrow.Table`).

#     If the `array` does not contain records at top-level, the Arrow table will consist
#     of one field whose name is `""`.

#     Arrow tables can maintain the distinction between "option-type but no elements are
#     missing" and "not option-type" at all levels, including the top level. However,
#     there is no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type.
#     Be aware of these type distinctions when passing data through Arrow or Parquet.

#     See also #ak.to_arrow, #ak.from_arrow, #ak.to_parquet.
#     """
#     pyarrow = _import_pyarrow("ak.to_arrow_table")

#     layout = to_layout(array, allow_record=False, allow_other=False)

#     if explode_records or isinstance(
#         ak._v2.operations.describe.type(layout), ak.types.RecordType
#     ):
#         names = layout.keys()
#         contents = [layout[name] for name in names]
#     else:
#         names = [""]
#         contents = [layout]

#     pa_arrays = []
#     pa_fields = []
#     for name, content in zip(names, contents):
#         pa_arrays.append(
#             to_arrow(
#                 content,
#                 list_to32=list_to32,
#                 string_to32=string_to32,
#                 bytestring_to32=bytestring_to32,
#             )
#         )
#         pa_fields.append(
#             pyarrow.field(name, pa_arrays[-1].type).with_nullable(
#                 isinstance(ak._v2.operations.describe.type(content), ak.types.OptionType)
#             )
#         )

#     batch = pyarrow.RecordBatch.from_arrays(pa_arrays, schema=pyarrow.schema(pa_fields))
#     return pyarrow.Table.from_batches([batch])
