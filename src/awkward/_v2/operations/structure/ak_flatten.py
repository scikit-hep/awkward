# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def flatten(array, axis=1, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data containing nested lists to flatten.
#         axis (None or int): If None, the operation flattens all levels of
#             nesting, returning a 1-dimensional array. Otherwise, it flattens
#             at a specified depth. The outermost dimension is `0`, followed
#             by `1`, etc., and negative values count backward from the
#             innermost: `-1` is the innermost dimension, `-2` is the next
#             level up, etc.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an array with one level of nesting removed by erasing the
#     boundaries between consecutive lists. Since this operates on a level of
#     nesting, `axis=0` is a special case that only removes values at the
#     top level that are equal to None.

#     Consider the following doubly nested `array`.

#         ak.Array([[
#                    [1.1, 2.2, 3.3],
#                    [],
#                    [4.4, 5.5],
#                    [6.6]],
#                   [],
#                   [
#                    [7.7],
#                    [8.8, 9.9]
#                   ]])

#     At `axis=1`, the outer lists (length 4, length 0, length 2) become a single
#     list (of length 6).

#         >>> print(ak.flatten(array, axis=1))
#         [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7], [8.8, 9.9]]

#     At `axis=2`, the inner lists (lengths 3, 0, 2, 1, 1, and 2) become three
#     lists (of lengths 6, 0, and 3).

#         >>> print(ak.flatten(array, axis=2))
#         [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [], [7.7, 8.8, 9.9]]

#     There's also an option to completely flatten the array with `axis=None`.
#     This is useful for passing the data to a function that doesn't care about
#     nested structure, such as a plotting routine.

#         >>> print(ak.flatten(array, axis=None))
#         [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

#     Missing values are eliminated by flattening: there is no distinction
#     between an empty list and a value of None at the level of flattening.

#         >>> array = ak.Array([[1.1, 2.2, 3.3], None, [4.4], [], [5.5]])
#         >>> ak.flatten(array, axis=1)
#         <Array [1.1, 2.2, 3.3, 4.4, 5.5] type='5 * float64'>

#     As a consequence, flattening at `axis=0` does only one thing: it removes
#     None values from the top level.

#         >>> ak.flatten(array, axis=0)
#         <Array [[1.1, 2.2, 3.3], [4.4], [], [5.5]] type='4 * var * float64'>

#     As a technical detail, the flattening operation can be trivial in a common
#     case, #ak.layout.ListOffsetArray in which the first `offset` is `0`.
#     In that case, the flattened data is simply the array node's `content`.

#         >>> array.layout
#         <ListOffsetArray64>
#             <offsets><Index64 i="[0 4 4 6]" offset="0" length="4"/></offsets>
#             <content><ListOffsetArray64>
#                 <offsets><Index64 i="[0 3 3 5 6 7 9]" offset="0" length="7"/></offsets>
#                 <content>
#                     <NumpyArray format="d" shape="9" data="1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9"/>
#                 </content>
#             </ListOffsetArray64></content>
#         </ListOffsetArray64>
#         >>> np.asarray(array.layout.content.content)
#         array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

#     However, it is important to keep in mind that this is a special case:
#     #ak.flatten and `content` are not interchangeable!
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     nplike = ak.nplike.of(layout)

#     if axis is None:
#         out = ak._v2._util.completely_flatten(layout)
#         assert isinstance(out, tuple) and all(isinstance(x, np.ndarray) for x in out)

#         if any(isinstance(x, nplike.ma.MaskedArray) for x in out):
#             out = ak._v2.contents.NumpyArray(nplike.ma.concatenate(out))
#         else:
#             out = ak._v2.contents.NumpyArray(nplike.concatenate(out))

#     elif axis == 0 or layout.axis_wrap_if_negative(axis) == 0:

#         def apply(layout):
#             if isinstance(layout, ak._v2._util.virtualtypes):
#                 return apply(layout.array)

#             elif isinstance(layout, ak._v2._util.unknowntypes):
#                 return apply(ak._v2.contents.NumpyArray(nplike.array([])))

#             elif isinstance(layout, ak._v2._util.indexedtypes):
#                 return apply(layout.project())

#             elif isinstance(layout, ak._v2._util.uniontypes):
#                 if not any(
#                     isinstance(x, ak._v2._util.optiontypes)
#                     and not isinstance(x, ak._v2.contents.UnmaskedArray)
#                     for x in layout.contents
#                 ):
#                     return layout

#                 tags = nplike.asarray(layout.tags)
#                 index = nplike.array(nplike.asarray(layout.index), copy=True)
#                 bigmask = nplike.empty(len(index), dtype=np.bool_)
#                 for tag, content in enumerate(layout.contents):
#                     if isinstance(content, ak._v2._util.optiontypes) and not isinstance(
#                         content, ak._v2.contents.UnmaskedArray
#                     ):
#                         bigmask[:] = False
#                         bigmask[tags == tag] = nplike.asarray(content.bytemask()).view(
#                             np.bool_
#                         )
#                         index[bigmask] = -1

#                 good = index >= 0
#                 return ak._v2.contents.UnionArray8_64(
#                     ak._v2.index.Index8(tags[good]),
#                     ak._v2.index.Index64(index[good]),
#                     layout.contents,
#                 )

#             elif isinstance(layout, ak._v2._util.optiontypes):
#                 return layout.project()

#             else:
#                 return layout

#         if isinstance(layout, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#             out = ak.partition.IrregularlyPartitionedArray(   # NO PARTITIONED ARRAY
#                 [apply(x) for x in layout.partitions]
#             )
#         else:
#             out = apply(layout)

#         return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)

#     else:
#         out = layout.flatten(axis)

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
