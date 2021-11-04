# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def packed(array, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Array whose internal structure will be packed.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an array with the same data as the input, but with packed inner structures:

#     - #ak.layout.NumpyArray becomes C-contiguous (if it isn't already)
#     - #ak.layout.RegularArray trims unreachable content
#     - #ak.layout.ListArray becomes #ak.layout.ListOffsetArray, making all list data contiguous
#     - #ak.layout.ListOffsetArray starts at `offsets[0] == 0`, trimming unreachable content
#     - #ak.layout.RecordArray trims unreachable contents
#     - #ak.layout.IndexedArray gets projected
#     - #ak.layout.IndexedOptionArray remains an #ak.layout.IndexedOptionArray (with simplified `index`) if it contains records, becomes #ak.layout.ByteMaskedArray otherwise
#     - #ak.layout.ByteMaskedArray becomes an #ak.layout.IndexedOptionArray if it contains records, stays a #ak.layout.ByteMaskedArray otherwise
#     - #ak.layout.BitMaskedArray becomes an #ak.layout.IndexedOptionArray if it contains records, stays a #ak.layout.BitMaskedArray otherwise
#     - #ak.layout.UnionArray gets projected contents
#     - #ak.layout.VirtualArray gets materialized
#     - #ak.layout.Record becomes a record over a single-item #ak.layout.RecordArray

#     Example:

#         >>> a = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])
#         >>> b = a[::-1]
#         >>> b
#         <Array [[7, 8, 9, 10], [6, ... [], [1, 2, 3]] type='5 * var * int64'>
#         >>> b.layout
#         <ListArray64>
#             <starts><Index64 i="[6 5 3 3 0]" offset="0" length="5" at="0x55e091c2b1f0"/></starts>
#             <stops><Index64 i="[10 6 5 3 3]" offset="0" length="5" at="0x55e091a6ce80"/></stops>
#             <content><NumpyArray format="l" shape="10" data="1 2 3 4 5 6 7 8 9 10" at="0x55e091c47260"/></content>
#         </ListArray64>
#         >>> c = ak.packed(b)
#         >>> c
#         <Array [[7, 8, 9, 10], [6, ... [], [1, 2, 3]] type='5 * var * int64'>
#         >>> c.layout
#         <ListOffsetArray64>
#             <offsets><Index64 i="[0 4 5 7 7 10]" offset="0" length="6" at="0x55e091b077a0"/></offsets>
#             <content><NumpyArray format="l" shape="10" data="7 8 9 10 6 4 5 1 2 3" at="0x55e091d04d30"/></content>
#         </ListOffsetArray64>

#     Performing these operations will minimize the output size of data sent to
#     #ak.to_buffers (though conversions through Arrow, #ak.to_arrow and
#     #ak.to_parquet, do not need this because packing is part of that conversion).

#     See also #ak.to_buffers.
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=True, allow_other=False
#     )

#     def transform(layout, depth=1, user=None):
#         return ak._v2._util.transform_child_layouts(
#             transform, _pack_layout(layout), depth, user
#         )

#     out = transform(layout)

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)


# def _pack_layout(layout):
#     nplike = ak.nplike.of(layout)

#     if isinstance(layout, ak._v2.contents.NumpyArray):
#         return layout.contiguous()

#     # EmptyArray is a no-op
#     elif isinstance(layout, ak._v2.contents.EmptyArray):
#         return layout

#     # Project indexed arrays
#     elif isinstance(layout, ak._v2._util.indexedoptiontypes):
#         if isinstance(layout.content, ak._v2._util.optiontypes):
#             return layout.simplify()

#         index = nplike.asarray(layout.index)
#         new_index = nplike.zeros_like(index)

#         is_none = index < 0
#         new_index[is_none] = -1
#         new_index[~is_none] = nplike.arange(len(new_index) - nplike.sum(is_none))

#         return ak._v2.contents.IndexedOptionArray64(
#             ak._v2.index.Index64(new_index),
#             layout.project(),
#             layout.identities,
#             layout.parameters,
#         )

#     # Project indexed arrays
#     elif isinstance(layout, ak._v2._util.indexedtypes):
#         return layout.project()

#     # ListArray performs both ordering and resizing
#     elif isinstance(
#         layout,
#         (
#             ak._v2.contents.ListArray32,
#             ak._v2.contents.ListArrayU32,
#             ak._v2.contents.ListArray64,
#         ),
#     ):
#         return layout.toListOffsetArray64(True)

#     # ListOffsetArray performs resizing
#     elif isinstance(
#         layout,
#         (
#             ak._v2.contents.ListOffsetArray32,
#             ak._v2.contents.ListOffsetArray64,
#             ak._v2.contents.ListOffsetArrayU32,
#         ),
#     ):
#         new_layout = layout.toListOffsetArray64(True)
#         new_length = new_layout.offsets[-1]
#         return ak._v2.contents.ListOffsetArray64(
#             new_layout.offsets,
#             new_layout.content[:new_length],
#             new_layout.identities,
#             new_layout.parameters,
#         )

#     # UnmaskedArray just wraps another array
#     elif isinstance(layout, ak._v2.contents.UnmaskedArray):
#         return ak._v2.contents.UnmaskedArray(
#             layout.content, layout.identities, layout.parameters
#         )

#     # UnionArrays can be simplified
#     # and their contents too
#     elif isinstance(layout, ak._v2._util.uniontypes):
#         layout = layout.simplify()

#         # If we managed to lose the drop type entirely
#         if not isinstance(layout, ak._v2._util.uniontypes):
#             return layout

#         # Pack simplified layout
#         tags = nplike.asarray(layout.tags)
#         index = nplike.asarray(layout.index)

#         new_contents = [None] * len(layout.contents)
#         new_index = nplike.zeros_like(index)

#         # Compact indices
#         for i in range(len(layout.contents)):
#             is_i = tags == i

#             new_contents[i] = layout.project(i)
#             new_index[is_i] = nplike.arange(nplike.sum(is_i))

#         return ak._v2.contents.UnionArray8_64(
#             ak._v2.index.Index8(tags),
#             ak._v2.index.Index64(new_index),
#             new_contents,
#             layout.identities,
#             layout.parameters,
#         )

#     # RecordArray contents can be truncated
#     elif isinstance(layout, ak._v2.contents.RecordArray):
#         return ak._v2.contents.RecordArray(
#             [c[: len(layout)] for c in layout.contents],
#             layout.recordlookup,
#             len(layout),
#             layout.identities,
#             layout.parameters,
#         )

#     # RegularArrays can change length
#     elif isinstance(layout, ak._v2.contents.RegularArray):
#         if not len(layout):
#             return layout

#         content = layout.content

#         # Truncate content to perfect multiple of the RegularArray size
#         if layout.size > 0:
#             r = len(content) % layout.size
#             content = content[: len(content) - r]
#         else:
#             content = content[:0]

#         return ak._v2.contents.RegularArray(
#             content,
#             layout.size,
#             len(layout),
#             layout.identities,
#             layout.parameters,
#         )

#     # BitMaskedArrays can change length
#     elif isinstance(layout, ak._v2.contents.BitMaskedArray):
#         layout = layout.simplify()

#         if not isinstance(ak.type(layout.content), ak._v2.types.PrimitiveType):
#             return layout.toIndexedOptionArray64()

#         return ak._v2.contents.BitMaskedArray(
#             layout.mask,
#             layout.content[: len(layout)],
#             layout.valid_when,
#             len(layout),
#             layout.lsb_order,
#             layout.identities,
#             layout.parameters,
#         )

#     # ByteMaskedArrays can change length
#     elif isinstance(layout, ak._v2.contents.ByteMaskedArray):
#         layout = layout.simplify()

#         if not isinstance(ak.type(layout.content), ak._v2.types.PrimitiveType):
#             return layout.toIndexedOptionArray64()

#         return ak._v2.contents.ByteMaskedArray(
#             layout.mask,
#             layout.content[: len(layout)],
#             layout.valid_when,
#             layout.identities,
#             layout.parameters,
#         )

#     elif isinstance(layout, ak._v2.contents.VirtualArray):
#         return layout.array

#     elif isinstance(layout, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#         return layout

#     elif isinstance(layout, ak._v2.record.Record):
#         return ak._v2.record.Record(layout.array[layout.at : layout.at + 1], 0)

#     # Finally, fall through to failure
#     else:
#         raise AssertionError(
#             "unrecognized layout: " + repr(layout)
#         )
