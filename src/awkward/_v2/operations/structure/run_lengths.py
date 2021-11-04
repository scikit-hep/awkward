# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def run_lengths(array, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data containing runs of numbers to count.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Computes the lengths of sequences of identical values at the deepest level
#     of nesting, returning an array with the same structure but with `int64` type.

#     For example,

#         >>> array = ak.Array([1.1, 1.1, 1.1, 2.2, 3.3, 3.3, 4.4, 4.4, 5.5])
#         >>> ak.run_lengths(array)
#         <Array [3, 1, 2, 2, 1] type='5 * int64'>

#     There are 3 instances of 1.1, followed by 1 instance of 2.2, 2 instances of 3.3,
#     2 instances of 4.4, and 1 instance of 5.5.

#     The order and uniqueness of the input data doesn't matter,

#         >>> array = ak.Array([1.1, 1.1, 1.1, 5.5, 4.4, 4.4, 1.1, 1.1, 5.5])
#         >>> ak.run_lengths(array)
#         <Array [3, 1, 2, 2, 1] type='5 * int64'>

#     just the difference between each value and its neighbors.

#     The data can be nested, but runs don't cross list boundaries.

#         >>> array = ak.Array([[1.1, 1.1, 1.1, 2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])
#         >>> ak.run_lengths(array)
#         <Array [[3, 1, 1], [1, 1], [1, 1]] type='3 * var * int64'>

#     This function recognizes strings as distinguishable values.

#         >>> array = ak.Array([["one", "one"], ["one", "two", "two"], ["three", "two", "two"]])
#         >>> ak.run_lengths(array)
#         <Array [[2], [1, 2], [1, 2]] type='3 * var * int64'>

#     Note that this can be combined with #ak.argsort and #ak.unflatten to compute
#     a "group by" operation:

#         >>> array = ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1},
#         ...                   {"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
#         >>> sorted = array[ak.argsort(array.x)]
#         >>> sorted.x
#         <Array [1, 1, 1, 2, 2, 3] type='6 * int64'>
#         >>> ak.run_lengths(sorted.x)
#         <Array [3, 2, 1] type='3 * int64'>
#         >>> ak.unflatten(sorted, ak.run_lengths(sorted.x)).tolist()
#         [[{'x': 1, 'y': 1.1}, {'x': 1, 'y': 1.1}, {'x': 1, 'y': 1.1}],
#          [{'x': 2, 'y': 2.2}, {'x': 2, 'y': 2.2}],
#          [{'x': 3, 'y': 3.3}]]

#     Unlike a database "group by," this operation can be applied in bulk to many sublists
#     (though the run lengths need to be fully flattened to be used as `counts` for
#     #ak.unflatten, and you need to specify `axis=-1` as the depth).

#         >>> array = ak.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1}],
#         ...                   [{"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]])
#         >>> sorted = array[ak.argsort(array.x)]
#         >>> sorted.x
#         <Array [[1, 1, 2], [1, 2, 3]] type='2 * var * int64'>
#         >>> ak.run_lengths(sorted.x)
#         <Array [[2, 1], [1, 1, 1]] type='2 * var * int64'>
#         >>> counts = ak.flatten(ak.run_lengths(sorted.x), axis=None)
#         >>> ak.unflatten(sorted, counts, axis=-1).tolist()
#         [[[{'x': 1, 'y': 1.1}, {'x': 1, 'y': 1.1}],
#           [{'x': 2, 'y': 2.2}]],
#          [[{'x': 1, 'y': 1.1}],
#           [{'x': 2, 'y': 2.2}],
#           [{'x': 3, 'y': 3.3}]]]

#     See also #ak.num, #ak.argsort, #ak.unflatten.
#     """
#     nplike = ak.nplike.of(array)

#     def lengths_of(data, offsets):
#         if len(data) == 0:
#             return nplike.empty(0, np.int64), offsets
#         else:
#             diffs = data[1:] != data[:-1]
#             if isinstance(diffs, ak._v2.highlevel.Array):
#                 diffs = nplike.asarray(diffs)
#             if offsets is not None:
#                 diffs[offsets[1:-1] - 1] = True
#             positions = nplike.nonzero(diffs)[0]
#             full_positions = nplike.empty(len(positions) + 2, np.int64)
#             full_positions[0] = 0
#             full_positions[-1] = len(data)
#             full_positions[1:-1] = positions + 1
#             nextcontent = full_positions[1:] - full_positions[:-1]
#             if offsets is None:
#                 nextoffsets = None
#             else:
#                 nextoffsets = nplike.searchsorted(full_positions, offsets, side="left")
#             return nextcontent, nextoffsets

#     def getfunction(layout):
#         if layout.branch_depth == (False, 1):
#             if isinstance(layout, ak._v2._util.indexedtypes):
#                 layout = layout.project()

#             if (
#                 layout.parameter("__array__") == "string"
#                 or layout.parameter("__array__") == "bytestring"
#             ):
#                 nextcontent, _ = lengths_of(ak._v2.highlevel.Array(layout), None)
#                 return lambda: ak._v2.contents.NumpyArray(nextcontent)

#             if not isinstance(layout, (ak._v2.contents.NumpyArray, ak._v2.contents.EmptyArray)):
#                 raise NotImplementedError(
#                     "run_lengths on "
#                     + type(layout).__name__
#
#                 )

#             nextcontent, _ = lengths_of(nplike.asarray(layout), None)
#             return lambda: ak._v2.contents.NumpyArray(nextcontent)

#         elif layout.branch_depth == (False, 2):
#             if isinstance(layout, ak._v2._util.indexedtypes):
#                 layout = layout.project()

#             if not isinstance(layout, ak._v2._util.listtypes):
#                 raise NotImplementedError(
#                     "run_lengths on "
#                     + type(layout).__name__
#
#                 )

#             if (
#                 layout.content.parameter("__array__") == "string"
#                 or layout.content.parameter("__array__") == "bytestring"
#             ):
#                 listoffsetarray = layout.toListOffsetArray64(False)
#                 offsets = nplike.asarray(listoffsetarray.offsets)
#                 content = listoffsetarray.content[offsets[0] : offsets[-1]]

#                 if isinstance(content, ak._v2._util.indexedtypes):
#                     content = content.project()

#                 nextcontent, nextoffsets = lengths_of(
#                     ak._v2.highlevel.Array(content), offsets - offsets[0]
#                 )
#                 return lambda: ak._v2.contents.ListOffsetArray64(
#                     ak._v2.index.Index64(nextoffsets), ak._v2.contents.NumpyArray(nextcontent)
#                 )

#             listoffsetarray = layout.toListOffsetArray64(False)
#             offsets = nplike.asarray(listoffsetarray.offsets)
#             content = listoffsetarray.content[offsets[0] : offsets[-1]]

#             if isinstance(content, ak._v2._util.indexedtypes):
#                 content = content.project()

#             if not isinstance(content, (ak._v2.contents.NumpyArray, ak._v2.contents.EmptyArray)):
#                 raise NotImplementedError(
#                     "run_lengths on "
#                     + type(layout).__name__
#                     + " with content "
#                     + type(content).__name__
#
#                 )

#             nextcontent, nextoffsets = lengths_of(
#                 nplike.asarray(content), offsets - offsets[0]
#             )
#             return lambda: ak._v2.contents.ListOffsetArray64(
#                 ak._v2.index.Index64(nextoffsets), ak._v2.contents.NumpyArray(nextcontent)
#             )

#         else:
#             return None

#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )

#     if isinstance(layout, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#         if len(layout.partitions) != 0 and layout.partitions[0].branch_depth == (
#             False,
#             1,
#         ):
#             out = ak._v2._util.recursively_apply(
#                 layout.toContent(),
#                 getfunction,
#                 pass_depth=False,
#                 pass_user=False,
#             )
#         else:
#             outparts = []
#             for part in layout.partitions:
#                 outparts.append(
#                     ak._v2._util.recursively_apply(
#                         part,
#                         getfunction,
#                         pass_depth=False,
#                         pass_user=False,
#                     )
#                 )
#             out = ak.partition.IrregularlyPartitionedArray(outparts)   # NO PARTITIONED ARRAY
#     else:
#         out = ak._v2._util.recursively_apply(
#             layout,
#             getfunction,
#             pass_depth=False,
#             pass_user=False,
#         )

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
