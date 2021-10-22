# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_arrow(array, highlevel=True, behavior=None):
    pass


#     """
#     Args:
#         array (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`,
#             or `pyarrow.Table`): Apache Arrow array to convert into an
#             Awkward Array.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Converts an Apache Arrow array into an Awkward Array.

#     Arrow arrays can maintain the distinction between "option-type but no elements are
#     missing" and "not option-type" at all levels except the top level. Arrow tables
#     can maintain the distinction at all levels. However, note that there is no distinction
#     between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type. Be aware of these
#     type distinctions when passing data through Arrow or Parquet.

#     See also #ak.to_arrow, #ak.to_arrow_table.
#     """
#     return _from_arrow(array, True, highlevel=highlevel, behavior=behavior)


# _pyarrow_to_numpy_dtype = {
#     "date32": (True, np.dtype("M8[D]")),
#     "date64": (False, np.dtype("M8[ms]")),
#     "date32[day]": (True, np.dtype("M8[D]")),
#     "date64[ms]": (False, np.dtype("M8[ms]")),
#     "time32[s]": (True, np.dtype("M8[s]")),
#     "time32[ms]": (True, np.dtype("M8[ms]")),
#     "time64[us]": (False, np.dtype("M8[us]")),
#     "time64[ns]": (False, np.dtype("M8[ns]")),
#     "timestamp[s]": (False, np.dtype("M8[s]")),
#     "timestamp[ms]": (False, np.dtype("M8[ms]")),
#     "timestamp[us]": (False, np.dtype("M8[us]")),
#     "timestamp[ns]": (False, np.dtype("M8[ns]")),
#     "duration[s]": (False, np.dtype("m8[s]")),
#     "duration[ms]": (False, np.dtype("m8[ms]")),
#     "duration[us]": (False, np.dtype("m8[us]")),
#     "duration[ns]": (False, np.dtype("m8[ns]")),
# }


# def _from_arrow(
#     array, pass_empty_field, struct_only=None, highlevel=True, behavior=None
# ):
#     pyarrow = _import_pyarrow("ak.from_arrow")

#     def popbuffers(array, tpe, buffers):
#         if isinstance(tpe, pyarrow.lib.DictionaryType):
#             index = popbuffers(array.indices, tpe.index_type, buffers)
#             content = handle_arrow(array.dictionary)

#             out = ak._v2.contents.IndexedArray32(
#                 ak._v2.index.Index32(index.content),
#                 content,
#                 parameters={"__array__": "categorical"},
#             ).simplify()

#             if isinstance(index, ak._v2.contents.BitMaskedArray):
#                 return ak._v2.contents.BitMaskedArray(
#                     index.mask,
#                     out,
#                     True,
#                     len(index),
#                     True,
#                     parameters={"__array__": "categorical"},
#                 ).simplify()
#             else:
#                 return out
#             # RETURNED because 'index' has already been offset-corrected.

#         elif isinstance(tpe, pyarrow.lib.StructType):
#             assert tpe.num_buffers == 1
#             mask = buffers.pop(0)
#             child_arrays = []
#             keys = []

#             if struct_only is None:
#                 for i in range(tpe.num_fields):
#                     content = popbuffers(array.field(tpe[i].name), tpe[i].type, buffers)
#                     if not tpe[i].nullable:
#                         content = content.content
#                     child_arrays.append(content)
#                     keys.append(tpe[i].name)
#             else:
#                 target = struct_only.pop()
#                 found = False
#                 for i in range(tpe.num_fields):
#                     if tpe[i].name == target:
#                         found = True
#                         content = popbuffers(
#                             array.field(tpe[i].name), tpe[i].type, buffers
#                         )
#                         if not tpe[i].nullable:
#                             content = content.content
#                         child_arrays.append(content)
#                         keys.append(tpe[i].name)
#                 assert found

#             out = ak._v2.contents.RecordArray(child_arrays, keys, length=len(array))
#             if mask is not None:
#                 mask = ak._v3.index.IndexU8(numpy.frombuffer(mask, dtype=np.uint8))
#                 return ak._v2.contents.BitMaskedArray(mask, out, True, len(out), True)
#             else:
#                 return ak._v2.contents.UnmaskedArray(out)
#             # RETURNED because each field has already been offset-corrected.

#         elif isinstance(tpe, pyarrow.lib.ListType):
#             assert tpe.num_buffers == 2
#             mask = buffers.pop(0)
#             offsets = ak._v2.index.Index32(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.int32)
#             )
#             content = popbuffers(array.values, tpe.value_type, buffers)
#             if not tpe.value_field.nullable:
#                 content = content.content

#             out = ak._v2.contents.ListOffsetArray32(offsets, content)
#             # No return yet!

#         elif isinstance(tpe, pyarrow.lib.LargeListType):
#             assert tpe.num_buffers == 2
#             mask = buffers.pop(0)
#             offsets = ak._v2.index.Index64(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.int64)
#             )
#             content = popbuffers(array.values, tpe.value_type, buffers)
#             if not tpe.value_field.nullable:
#                 content = content.content

#             out = ak._v2.contents.ListOffsetArray64(offsets, content)
#             # No return yet!

#         elif isinstance(tpe, pyarrow.lib.FixedSizeListType):
#             assert tpe.num_buffers == 1
#             mask = buffers.pop(0)
#             content = popbuffers(array.values, tpe.value_type, buffers)
#             if not tpe.value_field.nullable:
#                 content = content.content

#             out = ak._v2.contents.RegularArray(content, tpe.list_size)
#             # No return yet!

#         elif isinstance(tpe, pyarrow.lib.UnionType):
#             if tpe.mode == "sparse":
#                 assert tpe.num_buffers == 2
#             elif tpe.mode == "dense":
#                 assert tpe.num_buffers == 3
#             else:
#                 raise TypeError(
#                     "unrecognized Arrow union array mode: {0}".format(repr(tpe.mode))
#
#                 )

#             mask = buffers.pop(0)
#             tags = numpy.frombuffer(buffers.pop(0), dtype=np.int8)
#             if tpe.mode == "sparse":
#                 index = numpy.arange(len(tags), dtype=np.int32)
#             else:
#                 index = numpy.frombuffer(buffers.pop(0), dtype=np.int32)

#             contents = []
#             for i in range(tpe.num_fields):
#                 contents.append(popbuffers(array.field(i), tpe[i].type, buffers))
#             for i in range(len(contents)):
#                 these = index[tags == i]
#                 if len(these) == 0:
#                     contents[i] = contents[i][0:0]
#                 elif tpe.mode == "sparse":
#                     contents[i] = contents[i][: these[-1] + 1]
#                 else:
#                     contents[i] = contents[i][: these.max() + 1]
#             for i in range(len(contents)):
#                 if not tpe[i].nullable:
#                     contents[i] = contents[i].content

#             tags = ak._v2.index.Index8(tags)
#             index = ak._v2_index.Index32(index)
#             out = ak._v2.contents.UnionArray8_32(tags, index, contents)
#             # No return yet!

#         elif tpe == pyarrow.string():
#             assert tpe.num_buffers == 3
#             mask = buffers.pop(0)
#             offsets = ak._v2.index.Index32(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.int32)
#             )
#             contents = ak._v2.contents.NumpyArray(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
#             )
#             contents.setparameter("__array__", "char")
#             out = ak._v2.contents.ListOffsetArray32(offsets, contents)
#             out.setparameter("__array__", "string")
#             # No return yet!

#         elif tpe == pyarrow.large_string():
#             assert tpe.num_buffers == 3
#             mask = buffers.pop(0)
#             offsets = ak._v2.index.Index64(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.int64)
#             )
#             contents = ak._v2.contents.NumpyArray(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
#             )
#             contents.setparameter("__array__", "char")
#             out = ak._v2.contents.ListOffsetArray64(offsets, contents)
#             out.setparameter("__array__", "string")
#             # No return yet!

#         elif tpe == pyarrow.binary():
#             assert tpe.num_buffers == 3
#             mask = buffers.pop(0)
#             offsets = ak._v2.index.Index32(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.int32)
#             )
#             contents = ak._v2.contents.NumpyArray(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
#             )
#             contents.setparameter("__array__", "byte")
#             out = ak._v2.contents.ListOffsetArray32(offsets, contents)
#             out.setparameter("__array__", "bytestring")
#             # No return yet!

#         elif tpe == pyarrow.large_binary():
#             assert tpe.num_buffers == 3
#             mask = buffers.pop(0)
#             offsets = ak._v2.index.Index64(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.int64)
#             )
#             contents = ak._v2.contents.NumpyArray(
#                 numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
#             )
#             contents.setparameter("__array__", "byte")
#             out = ak._v2.contents.ListOffsetArray64(offsets, contents)
#             out.setparameter("__array__", "bytestring")
#             # No return yet!

#         elif tpe == pyarrow.bool_():
#             assert tpe.num_buffers == 2
#             mask = buffers.pop(0)
#             data = buffers.pop(0)
#             as_bytes = (
#                 numpy.unpackbits(numpy.frombuffer(data, dtype=np.uint8))
#                 .reshape(-1, 8)[:, ::-1]
#                 .reshape(-1)
#             )
#             out = ak._v2.contents.NumpyArray(as_bytes.view(np.bool_))
#             # No return yet!

#         elif isinstance(tpe, pyarrow.lib.DataType) and tpe.num_buffers == 1:
#             # This is a DataType(null)
#             mask = buffers.pop(0)
#             assert tpe.num_fields == 0
#             assert mask is None
#             out = ak._v2.contents.IndexedOptionArray64(
#                 ak._v2.index.Index64(numpy.full(len(array), -1, dtype=np.int64)),
#                 ak._v2.contents.EmptyArray(),
#             )
#             # No return yet!

#         elif isinstance(tpe, pyarrow.lib.DataType):
#             assert tpe.num_buffers == 2
#             mask = buffers.pop(0)
#             data = buffers.pop(0)

#             to64, dt = _pyarrow_to_numpy_dtype.get(str(tpe), (False, None))
#             if to64:
#                 data = numpy.frombuffer(data, dtype=np.int32).astype(np.int64)
#             if dt is None:
#                 dt = tpe.to_pandas_dtype()
#             out = ak._v2.contents.NumpyArray(numpy.frombuffer(data, dtype=dt))
#             # No return yet!

#         else:
#             raise TypeError(
#                 "unrecognized Arrow array type: {0}".format(repr(tpe))
#
#             )

#         # All 'no return yet' cases need to become option-type (even if the UnmaskedArray
#         # is just going to get stripped off in the recursive step that calls this one).
#         if mask is not None:
#             mask = ak._v2.index.IndexU8(numpy.frombuffer(mask, dtype=np.uint8))
#             out = ak._v2.contents.BitMaskedArray(mask, out, True, len(out), True)
#         else:
#             out = ak._v2.contents.UnmaskedArray(out)

#         # All 'no return yet' cases need to be corrected for pyarrow's 'offset'.
#         if array.offset == 0 and len(array) == len(out):
#             return out
#         else:
#             return out[array.offset : array.offset + len(array)]

#     def handle_arrow(obj):
#         if isinstance(obj, pyarrow.lib.Array):
#             buffers = obj.buffers()
#             out = popbuffers(obj, obj.type, buffers)
#             assert len(buffers) == 0
#             if isinstance(out, ak._v2.contents.UnmaskedArray):
#                 return out.content
#             else:
#                 return out

#         elif isinstance(obj, pyarrow.lib.ChunkedArray):
#             layouts = [handle_arrow(x) for x in obj.chunks if len(x) > 0]
#             if all(isinstance(x, ak._v2.contents.UnmaskedArray) for x in layouts):
#                 layouts = [x.content for x in layouts]
#             if len(layouts) == 1:
#                 return layouts[0]
#             else:
#                 return ak.operations.structure.concatenate(layouts, highlevel=False)

#         elif isinstance(obj, pyarrow.lib.RecordBatch):
#             child_array = []
#             for i in range(obj.num_columns):
#                 layout = handle_arrow(obj.column(i))
#                 if obj.schema.field(i).nullable and not isinstance(
#                     layout, ak._v2._util.optiontypes
#                 ):
#                     layout = ak._v2.contents.UnmaskedArray(layout)
#                 child_array.append(layout)
#             if pass_empty_field and list(obj.schema.names) == [""]:
#                 return child_array[0]
#             else:
#                 return ak._v2.contents.RecordArray(child_array, obj.schema.names)

#         elif isinstance(obj, pyarrow.lib.Table):
#             batches = obj.combine_chunks().to_batches()
#             if len(batches) == 0:
#                 # zero-length array with the right type
#                 return from_buffers(_parquet_schema_to_form(obj.schema), 0, {})
#             elif len(batches) == 1:
#                 return handle_arrow(batches[0])
#             else:
#                 arrays = [handle_arrow(batch) for batch in batches if len(batch) > 0]
#                 return ak.operations.structure.concatenate(arrays, highlevel=False)

#         elif (
#             isinstance(obj, Iterable)
#             and isinstance(obj, Sized)
#             and len(obj) > 0
#             and all(isinstance(x, pyarrow.lib.RecordBatch) for x in obj)
#             and any(len(x) > 0 for x in obj)
#         ):
#             chunks = []
#             for batch in obj:
#                 chunk = handle_arrow(batch)
#                 if len(chunk) > 0:
#                     chunks.append(chunk)
#             if len(chunks) == 1:
#                 return chunks[0]
#             else:
#                 return ak.operations.structure.concatenate(chunks, highlevel=False)

#         else:
#             raise TypeError(
#                 "unrecognized Arrow type: {0}".format(type(obj))
#
#             )

#     return ak._v2._util.maybe_wrap(handle_arrow(array), behavior, highlevel)
