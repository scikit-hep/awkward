# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import distutils

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def _import_pyarrow(name):  # move this to _util
    try:
        import pyarrow
    except ImportError:
        raise ImportError(
            """to use {0}, you must install pyarrow:

    pip install pyarrow

or

    conda install -c conda-forge pyarrow
""".format(
                name
            )
        )
    else:
        if distutils.version.LooseVersion(
            pyarrow.__version__
        ) < distutils.version.LooseVersion("5.0.0"):
            raise ImportError("pyarrow 5.0.0 or later required for {0}".format(name))
        return pyarrow


def to_arrow(
    array, list_to32=False, string_to32=True, bytestring_to32=True, allow_tensor=True
):
    pass


#     """
#     Args:
#         array: Data to convert to an Apache Arrow array.
#         list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
#             if they're small enough, even if it means an extra conversion. Otherwise,
#             signed 32-bit #ak.layout.ListOffsetArray maps to Arrow `ListType` and
#             all others map to Arrow `LargeListType`.
#         string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
#         bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
#         allow_tensor (bool): If True, convert regular-length lists to `pyarrow.lib.Tensor`;
#             otherwise, make `pyarrow.lib.ListArray` (generating offsets). This is used
#             by #ak.to_parquet, since Parquet files can't contain regular-length tensors.

#     Converts an Awkward Array into an Apache Arrow array.

#     This produces arrays of type `pyarrow.Array`. You might need to further
#     manipulations (using the pyarrow library) to build a `pyarrow.ChunkedArray`,
#     a `pyarrow.RecordBatch`, or a `pyarrow.Table`.

#     Arrow arrays can maintain the distinction between "option-type but no elements are
#     missing" and "not option-type" at all levels except the top level. Also, there is
#     no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type. Be
#     aware of these type distinctions when passing data through Arrow or Parquet.

#     See also #ak.from_arrow, #ak.to_arrow_table, #ak.to_parquet.
#     """
#     pyarrow = _import_pyarrow("ak.to_arrow")

#     layout = to_layout(array, allow_record=False, allow_other=False)

#     def recurse(layout, mask, is_option):
#         if isinstance(layout, ak._v2.contents.NumpyArray):
#             numpy_arr = numpy.asarray(layout)
#             length = len(numpy_arr)
#             arrow_type = pyarrow.from_numpy_dtype(numpy_arr.dtype)

#             if issubclass(numpy_arr.dtype.type, (bool, np.bool_)):
#                 if numpy_arr.ndim == 1:
#                     if len(numpy_arr) % 8 == 0:
#                         ready_to_pack = numpy_arr
#                     else:
#                         ready_to_pack = numpy.empty(
#                             int(numpy.ceil(len(numpy_arr) / 8.0)) * 8,
#                             dtype=numpy_arr.dtype,
#                         )
#                         ready_to_pack[: len(numpy_arr)] = numpy_arr
#                         ready_to_pack[len(numpy_arr) :] = 0
#                     numpy_arr = numpy.packbits(
#                         ready_to_pack.reshape(-1, 8)[:, ::-1].reshape(-1)
#                     )
#                 else:
#                     return recurse(
#                         from_numpy(numpy_arr, regulararray=True, highlevel=False),
#                         mask,
#                         is_option,
#                     )

#             if numpy_arr.ndim == 1:
#                 if mask is not None:
#                     return pyarrow.Array.from_buffers(
#                         arrow_type,
#                         length,
#                         [pyarrow.py_buffer(mask), pyarrow.py_buffer(numpy_arr)],
#                     )
#                 else:
#                     return pyarrow.Array.from_buffers(
#                         arrow_type, length, [None, pyarrow.py_buffer(numpy_arr)]
#                     )
#             elif allow_tensor:
#                 return pyarrow.Tensor.from_numpy(numpy_arr)
#             else:
#                 return recurse(
#                     from_numpy(numpy_arr, regulararray=True, highlevel=False),
#                     mask,
#                     is_option,
#                 )

#         elif isinstance(layout, ak._v2.contents.EmptyArray):
#             return pyarrow.Array.from_buffers(pyarrow.null(), 0, [None])

#         elif isinstance(layout, ak._v2.contents.ListOffsetArray32):
#             offsets = numpy.asarray(layout.offsets, dtype=np.int32)

#             if layout.parameter("__array__") == "bytestring":
#                 if mask is None:
#                     arrow_arr = pyarrow.Array.from_buffers(
#                         pyarrow.binary(),
#                         len(offsets) - 1,
#                         [
#                             None,
#                             pyarrow.py_buffer(offsets),
#                             pyarrow.py_buffer(layout.content),
#                         ],
#                         children=[],
#                     )
#                 else:
#                     arrow_arr = pyarrow.Array.from_buffers(
#                         pyarrow.binary(),
#                         len(offsets) - 1,
#                         [
#                             pyarrow.py_buffer(mask),
#                             pyarrow.py_buffer(offsets),
#                             pyarrow.py_buffer(layout.content),
#                         ],
#                         children=[],
#                     )
#                 return arrow_arr

#             if layout.parameter("__array__") == "string":
#                 if mask is None:
#                     arrow_arr = pyarrow.StringArray.from_buffers(
#                         len(offsets) - 1,
#                         pyarrow.py_buffer(offsets),
#                         pyarrow.py_buffer(layout.content),
#                     )
#                 else:
#                     arrow_arr = pyarrow.StringArray.from_buffers(
#                         len(offsets) - 1,
#                         pyarrow.py_buffer(offsets),
#                         pyarrow.py_buffer(layout.content),
#                         pyarrow.py_buffer(mask),
#                     )
#                 return arrow_arr

#             content_buffer = recurse(layout.content[: offsets[-1]], None, False)
#             content_type = pyarrow.list_(content_buffer.type).value_field.with_nullable(
#                 isinstance(
#                     ak.operations.describe.type(layout.content), ak.types.OptionType
#                 )
#             )
#             if mask is None:
#                 arrow_arr = pyarrow.Array.from_buffers(
#                     pyarrow.list_(content_type),
#                     len(offsets) - 1,
#                     [None, pyarrow.py_buffer(offsets)],
#                     children=[content_buffer],
#                 )
#             else:
#                 arrow_arr = pyarrow.Array.from_buffers(
#                     pyarrow.list_(content_type),
#                     len(offsets) - 1,
#                     [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)],
#                     children=[content_buffer],
#                 )
#             return arrow_arr

#         elif isinstance(
#             layout,
#             (ak._v2.contents.ListOffsetArray64, ak._v2.contents.ListOffsetArrayU32),
#         ):
#             if layout.parameter("__array__") == "bytestring":
#                 downsize = bytestring_to32
#             elif layout.parameter("__array__") == "string":
#                 downsize = string_to32
#             else:
#                 downsize = list_to32

#             offsets = numpy.asarray(layout.offsets)

#             if downsize and offsets[-1] <= np.iinfo(np.int32).max:
#                 small_layout = ak._v2.contents.ListOffsetArray32(
#                     ak._v2.index.Index32(offsets.astype(np.int32)),
#                     layout.content,
#                     parameters=layout.parameters,
#                 )
#                 return recurse(small_layout, mask, is_option)

#             offsets = numpy.asarray(layout.offsets, dtype=np.int64)

#             if layout.parameter("__array__") == "bytestring":
#                 if mask is None:
#                     arrow_arr = pyarrow.Array.from_buffers(
#                         pyarrow.large_binary(),
#                         len(offsets) - 1,
#                         [
#                             None,
#                             pyarrow.py_buffer(offsets),
#                             pyarrow.py_buffer(layout.content),
#                         ],
#                         children=[],
#                     )
#                 else:
#                     arrow_arr = pyarrow.Array.from_buffers(
#                         pyarrow.large_binary(),
#                         len(offsets) - 1,
#                         [
#                             pyarrow.py_buffer(mask),
#                             pyarrow.py_buffer(offsets),
#                             pyarrow.py_buffer(layout.content),
#                         ],
#                         children=[],
#                     )
#                 return arrow_arr

#             if layout.parameter("__array__") == "string":
#                 if mask is None:
#                     arrow_arr = pyarrow.LargeStringArray.from_buffers(
#                         len(offsets) - 1,
#                         pyarrow.py_buffer(offsets),
#                         pyarrow.py_buffer(layout.content),
#                     )
#                 else:
#                     arrow_arr = pyarrow.LargeStringArray.from_buffers(
#                         len(offsets) - 1,
#                         pyarrow.py_buffer(offsets),
#                         pyarrow.py_buffer(layout.content),
#                         pyarrow.py_buffer(mask),
#                     )
#                 return arrow_arr

#             content_buffer = recurse(layout.content[: offsets[-1]], None, False)
#             content_type = pyarrow.list_(content_buffer.type).value_field.with_nullable(
#                 isinstance(
#                     ak.operations.describe.type(layout.content), ak.types.OptionType
#                 )
#             )
#             if mask is None:
#                 arrow_arr = pyarrow.Array.from_buffers(
#                     pyarrow.large_list(content_type),
#                     len(offsets) - 1,
#                     [None, pyarrow.py_buffer(offsets)],
#                     children=[content_buffer],
#                 )
#             else:
#                 arrow_arr = pyarrow.Array.from_buffers(
#                     pyarrow.large_list(content_type),
#                     len(offsets) - 1,
#                     [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)],
#                     children=[content_buffer],
#                 )
#             return arrow_arr

#         elif isinstance(layout, ak._v2.contents.RegularArray):
#             return recurse(
#                 layout.broadcast_tooffsets64(layout.compact_offsets64()),
#                 mask,
#                 is_option,
#             )

#         elif isinstance(
#             layout,
#             (
#                 ak._v2.contents.ListArray32,
#                 ak._v2.contents.ListArrayU32,
#                 ak._v2.contents.ListArray64,
#             ),
#         ):
#             return recurse(
#                 layout.broadcast_tooffsets64(layout.compact_offsets64()),
#                 mask,
#                 is_option,
#             )

#         elif isinstance(layout, ak._v2.contents.RecordArray):
#             values = [
#                 recurse(x[: len(layout)], mask, is_option) for x in layout.contents
#             ]

#             min_list_len = min(map(len, values))

#             types = pyarrow.struct(
#                 [
#                     pyarrow.field(layout.key(i), values[i].type).with_nullable(
#                         isinstance(ak.operations.describe.type(x), ak.types.OptionType)
#                     )
#                     for i, x in enumerate(layout.contents)
#                 ]
#             )

#             if mask is not None:
#                 return pyarrow.Array.from_buffers(
#                     types, min_list_len, [pyarrow.py_buffer(mask)], children=values
#                 )
#             else:
#                 return pyarrow.Array.from_buffers(
#                     types, min_list_len, [None], children=values
#                 )

#         elif isinstance(
#             layout,
#             (
#                 ak._v2.contents.UnionArray8_32,
#                 ak._v2.contents.UnionArray8_64,
#                 ak._v2.contents.UnionArray8_U32,
#             ),
#         ):
#             tags = numpy.asarray(layout.tags)
#             index = numpy.asarray(layout.index)
#             copied_index = False
#             if mask is not None:
#                 bytemask = (
#                     numpy.unpackbits(mask)
#                     .reshape(-1, 8)[:, ::-1]
#                     .reshape(-1)
#                     .view(np.bool_)
#                 )[: len(tags)]

#             values = []
#             for tag, content in enumerate(layout.contents):
#                 selected_tags = tags == tag
#                 this_index = index[selected_tags]
#                 if mask is not None:
#                     length = int(numpy.ceil(len(this_index) / 8.0)) * 8
#                     if len(numpy.unique(this_index)) == len(this_index):
#                         this_bytemask = numpy.zeros(length, dtype=np.uint8)
#                         this_bytemask[this_index] = bytemask[selected_tags]
#                     else:
#                         this_bytemask = numpy.empty(length, dtype=np.uint8)
#                         this_bytemask[: len(this_index)] = bytemask[selected_tags]
#                         this_bytemask[len(this_index) :] = 0

#                         content = content[this_index]
#                         this_index = numpy.arange(len(this_index))
#                         if not copied_index:
#                             copied_index = True
#                             index = numpy.array(index, copy=True)
#                         index[selected_tags] = this_index

#                     this_mask = numpy.packbits(
#                         this_bytemask.reshape(-1, 8)[:, ::-1].reshape(-1)
#                     )

#                 else:
#                     this_mask = None

#                 values.append(recurse(content, this_mask, is_option))

#             types = pyarrow.union(
#                 [
#                     pyarrow.field(str(i), values[i].type).with_nullable(
#                         is_option
#                         or isinstance(layout.content(i).type, ak.types.OptionType)
#                     )
#                     for i in range(len(values))
#                 ],
#                 "dense",
#                 list(range(len(values))),
#             )

#             return pyarrow.Array.from_buffers(
#                 types,
#                 len(layout.tags),
#                 [
#                     None,
#                     pyarrow.py_buffer(tags),
#                     pyarrow.py_buffer(index.astype(np.int32)),
#                 ],
#                 children=values,
#             )

#         elif isinstance(
#             layout,
#             (
#                 ak._v2.contents.IndexedArray32,
#                 ak._v2.contents.IndexedArrayU32,
#                 ak._v2.contents.IndexedArray64,
#             ),
#         ):
#             index = numpy.asarray(layout.index)

#             if layout.parameter("__array__") == "categorical":
#                 dictionary = recurse(layout.content, None, False)
#                 if mask is None:
#                     return pyarrow.DictionaryArray.from_arrays(index, dictionary)
#                 else:
#                     bytemask = (
#                         numpy.unpackbits(~mask)
#                         .reshape(-1, 8)[:, ::-1]
#                         .reshape(-1)
#                         .view(np.bool_)
#                     )[: len(index)]
#                     return pyarrow.DictionaryArray.from_arrays(
#                         index, dictionary, bytemask
#                     )

#             else:
#                 layout_content = layout.content

#                 if len(layout_content) == 0:
#                     empty = recurse(layout_content, None, False)
#                     if mask is None:
#                         return empty
#                     else:
#                         return pyarrow.array([None] * len(index)).cast(empty.type)

#                 elif isinstance(layout_content, ak._v2.contents.RecordArray):
#                     values = [
#                         recurse(x[: len(layout_content)][index], mask, is_option)
#                         for x in layout_content.contents
#                     ]

#                     min_list_len = min(map(len, values))

#                     types = pyarrow.struct(
#                         [
#                             pyarrow.field(
#                                 layout_content.key(i), values[i].type
#                             ).with_nullable(
#                                 isinstance(
#                                     ak.operations.describe.type(x), ak.types.OptionType
#                                 )
#                             )
#                             for i, x in enumerate(layout_content.contents)
#                         ]
#                     )

#                     if mask is not None:
#                         return pyarrow.Array.from_buffers(
#                             types,
#                             min_list_len,
#                             [pyarrow.py_buffer(mask)],
#                             children=values,
#                         )
#                     else:
#                         return pyarrow.Array.from_buffers(
#                             types, min_list_len, [None], children=values
#                         )

#                 else:
#                     return recurse(layout_content[index], mask, is_option)

#         elif isinstance(
#             layout,
#             (ak._v2.contents.IndexedOptionArray32, ak._v2.contents.IndexedOptionArray64),
#         ):
#             index = numpy.array(layout.index, copy=True)
#             nulls = index < 0
#             index[nulls] = 0

#             if layout.parameter("__array__") == "categorical":
#                 dictionary = recurse(layout.content, None, False)

#                 if mask is None:
#                     bytemask = nulls
#                 else:
#                     bytemask = (
#                         numpy.unpackbits(~mask)
#                         .reshape(-1, 8)[:, ::-1]
#                         .reshape(-1)
#                         .view(np.bool_)
#                     )[: len(index)]
#                     bytemask[nulls] = True

#                 return pyarrow.DictionaryArray.from_arrays(index, dictionary, bytemask)

#             else:
#                 if len(nulls) % 8 == 0:
#                     this_bytemask = (~nulls).view(np.uint8)
#                 else:
#                     length = int(numpy.ceil(len(nulls) / 8.0)) * 8
#                     this_bytemask = numpy.empty(length, dtype=np.uint8)
#                     this_bytemask[: len(nulls)] = ~nulls
#                     this_bytemask[len(nulls) :] = 0

#                 this_bitmask = numpy.packbits(
#                     this_bytemask.reshape(-1, 8)[:, ::-1].reshape(-1)
#                 )

#                 if isinstance(layout, ak._v2.contents.IndexedOptionArray32):
#                     next = ak._v2.contents.IndexedArray32(
#                         ak._v2.index.Index32(index), layout.content
#                     )
#                 else:
#                     next = ak._v2.contents.IndexedArray64(
#                         ak._v2.index.Index64(index), layout.content
#                     )

#                 if mask is None:
#                     return recurse(next, this_bitmask, True)
#                 else:
#                     return recurse(next, mask & this_bitmask, True)

#         elif isinstance(layout, ak._v2.contents.BitMaskedArray):
#             bitmask = numpy.asarray(layout.mask, dtype=np.uint8)

#             if layout.lsb_order is False:
#                 bitmask = numpy.packbits(
#                     numpy.unpackbits(bitmask).reshape(-1, 8)[:, ::-1].reshape(-1)
#                 )

#             if layout.valid_when is False:
#                 bitmask = ~bitmask

#             return recurse(layout.content[: len(layout)], bitmask, True).slice(
#                 length=min(len(bitmask) * 8, len(layout.content))
#             )

#         elif isinstance(layout, ak._v2.contents.ByteMaskedArray):
#             mask = numpy.asarray(layout.mask, dtype=np.bool_) == layout.valid_when

#             bytemask = numpy.zeros(
#                 8 * math.ceil(len(layout.content) / 8), dtype=np.bool_
#             )
#             bytemask[: len(mask)] = mask
#             bytemask[len(mask) :] = 0
#             bitmask = numpy.packbits(bytemask.reshape(-1, 8)[:, ::-1].reshape(-1))

#             return recurse(layout.content[: len(layout)], bitmask, True).slice(
#                 length=len(mask)
#             )

#         elif isinstance(layout, (ak._v2.contents.UnmaskedArray)):
#             return recurse(layout.content, None, True)

#         elif isinstance(layout, (ak._v2.contents.VirtualArray)):
#             return recurse(layout.array, None, False)

#         elif isinstance(layout, (ak.partition.PartitionedArray)):
#             return pyarrow.chunked_array(
#                 [recurse(x, None, False) for x in layout.partitions]
#             )

#         else:
#             raise TypeError(
#                 "unrecognized array type: {0}".format(repr(layout))
#
#             )

#     return recurse(layout, None, False)
