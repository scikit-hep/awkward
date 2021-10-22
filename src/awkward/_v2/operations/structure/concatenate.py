# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._connect._numpy.implements("concatenate")
def concatenate(
    arrays, axis=0, merge=True, mergebool=True, highlevel=True, behavior=None
):
    pass


#     """
#     Args:
#         arrays: Arrays to concatenate along any dimension.
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         merge (bool): If True, combine data into the same buffers wherever
#             possible, eliminating unnecessary #ak.layout.UnionArray8_64 types
#             at the expense of materializing #ak.layout.VirtualArray nodes.
#         mergebool (bool): If True, boolean and nummeric data can be combined
#             into the same buffer, losing information about False vs `0` and
#             True vs `1`; otherwise, they are kept in separate buffers with
#             distinct types (using an #ak.layout.UnionArray8_64).
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an array with `arrays` concatenated. For `axis=0`, this means that
#     one whole array follows another. For `axis=1`, it means that the `arrays`
#     must have the same lengths and nested lists are each concatenated,
#     element for element, and similarly for deeper levels.
#     """
#     contents = [
#         ak.operations.convert.to_layout(
#             x, allow_record=False if axis == 0 else True, allow_other=True
#         )
#         for x in arrays
#     ]
#     if not any(
#         isinstance(
#             x,
#             (ak._v2.contents.Content, ak.partition.PartitionedArray, ak._v2.contents.Content),
#         )
#         for x in contents
#     ):
#         raise ValueError(
#             "need at least one array to concatenate"
#             + ak._v2._util.exception_suffix(__file__)
#         )

#     first_content = [
#         x
#         for x in contents
#         if isinstance(
#             x,
#             (ak._v2.contents.Content, ak.partition.PartitionedArray, ak._v2.contents.Content),
#         )
#     ][0]
#     posaxis = first_content.axis_wrap_if_negative(axis)
#     maxdepth = max(
#         [
#             x.minmax_depth[1]
#             for x in contents
#             if isinstance(
#                 x,
#                 (
#                     ak._v2.contents.Content,
#                     ak.partition.PartitionedArray,
#                     ak._v2.contents.Content,
#                 ),
#             )
#         ]
#     )
#     if not 0 <= posaxis < maxdepth:
#         raise ValueError(
#             "axis={0} is beyond the depth of this array or the depth of this array "
#             "is ambiguous".format(axis) + ak._v2._util.exception_suffix(__file__)
#         )
#     for x in contents:
#         if isinstance(x, ak._v2.contents.Content):
#             if x.axis_wrap_if_negative(axis) != posaxis:
#                 raise ValueError(
#                     "arrays to concatenate do not have the same depth for negative "
#                     "axis={0}".format(axis) + ak._v2._util.exception_suffix(__file__)
#                 )

#     if any(isinstance(x, ak.partition.PartitionedArray) for x in contents):
#         if posaxis == 0:
#             partitions = []
#             offsets = [0]
#             for content in contents:
#                 if isinstance(content, ak.partition.PartitionedArray):
#                     start = 0
#                     for stop, part in __builtins__["zip"](
#                         content.stops, content.partitions
#                     ):
#                         count = stop - start
#                         start = stop
#                         partitions.append(part)
#                         offsets.append(offsets[-1] + count)
#                 elif isinstance(content, ak._v2.contents.Content):
#                     partitions.append(content)
#                     offsets.append(offsets[-1] + len(content))
#                 else:
#                     partitions.append(
#                         ak.operations.convert.from_iter([content], highlevel=False)
#                     )
#                     offsets.append(offsets[-1] + 1)

#             out = ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])

#         else:
#             for content in contents:
#                 if isinstance(content, ak.partition.PartitionedArray):
#                     stops = content.stops
#                     slices = []
#                     start = 0
#                     for stop in stops:
#                         slices.append(slice(start, stop))
#                         start = stop
#                     break

#             partitions = []
#             offsets = [0]
#             for slc in slices:
#                 newcontents = []
#                 for content in contents:
#                     if isinstance(content, ak.partition.PartitionedArray):
#                         newcontents.append(content[slc].toContent())
#                     elif isinstance(content, ak._v2.contents.Content):
#                         newcontents.append(content[slc])
#                     else:
#                         newcontents.append(content)

#                 partitions.append(
#                     concatenate(
#                         newcontents,
#                         axis=axis,
#                         merge=merge,
#                         mergebool=mergebool,
#                         highlevel=False,
#                     )
#                 )
#                 offsets.append(offsets[-1] + len(partitions[-1]))

#             out = ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])

#     elif posaxis == 0:
#         contents = [
#             x
#             if isinstance(x, ak._v2.contents.Content)
#             else ak.operations.convert.to_layout([x])
#             for x in contents
#         ]
#         batch = [contents[0]]
#         for x in contents[1:]:
#             if batch[-1].mergeable(x, mergebool=mergebool):
#                 batch.append(x)
#             else:
#                 collapsed = batch[0].mergemany(batch[1:])
#                 batch = [collapsed.merge_as_union(x)]

#         out = batch[0].mergemany(batch[1:])
#         if isinstance(out, ak._v2._util.uniontypes):
#             out = out.simplify(merge=merge, mergebool=mergebool)

#     else:

#         def getfunction(inputs, depth):
#             if depth == posaxis and any(
#                 isinstance(x, ak._v2._util.optiontypes) for x in inputs
#             ):
#                 nextinputs = []
#                 for x in inputs:
#                     if isinstance(x, ak._v2._util.optiontypes) and isinstance(
#                         x.content, ak._v2._util.listtypes
#                     ):
#                         nextinputs.append(fill_none(x, [], axis=0, highlevel=False))
#                     else:
#                         nextinputs.append(x)
#                 inputs = nextinputs

#             if depth == posaxis and all(
#                 isinstance(x, ak._v2._util.listtypes)
#                 or (isinstance(x, ak._v2.contents.NumpyArray) and x.ndim > 1)
#                 or not isinstance(x, ak._v2.contents.Content)
#                 for x in inputs
#             ):
#                 nplike = ak.nplike.of(*inputs)

#                 length = max(
#                     [len(x) for x in inputs if isinstance(x, ak._v2.contents.Content)]
#                 )
#                 nextinputs = []
#                 for x in inputs:
#                     if isinstance(x, ak._v2.contents.Content):
#                         nextinputs.append(x)
#                     else:
#                         nextinputs.append(
#                             ak._v2.contents.ListOffsetArray64(
#                                 ak._v2.index.Index64(
#                                     nplike.arange(length + 1, dtype=np.int64)
#                                 ),
#                                 ak._v2.contents.NumpyArray(
#                                     nplike.broadcast_to(nplike.array([x]), (length,))
#                                 ),
#                             )
#                         )

#                 counts = nplike.zeros(len(nextinputs[0]), dtype=np.int64)
#                 all_counts = []
#                 all_flatten = []
#                 for x in nextinputs:
#                     o, f = x.offsets_and_flatten(1)
#                     o = nplike.asarray(o)
#                     c = o[1:] - o[:-1]
#                     nplike.add(counts, c, out=counts)
#                     all_counts.append(c)
#                     all_flatten.append(f)

#                 offsets = nplike.empty(len(nextinputs[0]) + 1, dtype=np.int64)
#                 offsets[0] = 0
#                 nplike.cumsum(counts, out=offsets[1:])

#                 offsets = ak._v2.index.Index64(offsets)
#                 tags, index = ak._v2.contents.UnionArray8_64.nested_tags_index(
#                     offsets,
#                     [ak._v2.index.Index64(x) for x in all_counts],
#                 )
#                 inner = ak._v2.contents.UnionArray8_64(tags, index, all_flatten)

#                 out = ak._v2.contents.ListOffsetArray64(
#                     offsets, inner.simplify(merge=merge, mergebool=mergebool)
#                 )
#                 return lambda: (out,)

#             elif any(
#                 x.minmax_depth == (1, 1)
#                 for x in inputs
#                 if isinstance(x, ak._v2.contents.Content)
#             ):
#                 raise ValueError(
#                     "at least one array is not deep enough to concatenate at "
#                     "axis={0}".format(axis) + ak._v2._util.exception_suffix(__file__)
#                 )

#             else:
#                 return None

#         out = ak._v2._util.broadcast_and_apply(
#             contents,
#             getfunction,
#             behavior=ak._v2._util.behaviorof(*arrays, behavior=behavior),
#             allow_records=True,
#             right_broadcast=False,
#             pass_depth=True,
#         )[0]

#     return ak._v2._util.maybe_wrap(
#         out, ak._v2._util.behaviorof(*arrays, behavior=behavior), highlevel
#     )
# @ak._connect._numpy.implements("concatenate")
# def concatenate(
#     arrays, axis=0, merge=True, mergebool=True, highlevel=True, behavior=None
# ):
#     """
#     Args:
#         arrays: Arrays to concatenate along any dimension.
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         merge (bool): If True, combine data into the same buffers wherever
#             possible, eliminating unnecessary #ak.layout.UnionArray8_64 types
#             at the expense of materializing #ak.layout.VirtualArray nodes.
#         mergebool (bool): If True, boolean and nummeric data can be combined
#             into the same buffer, losing information about False vs `0` and
#             True vs `1`; otherwise, they are kept in separate buffers with
#             distinct types (using an #ak.layout.UnionArray8_64).
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an array with `arrays` concatenated. For `axis=0`, this means that
#     one whole array follows another. For `axis=1`, it means that the `arrays`
#     must have the same lengths and nested lists are each concatenated,
#     element for element, and similarly for deeper levels.
#     """
#     contents = [
#         ak.operations.convert.to_layout(
#             x, allow_record=False if axis == 0 else True, allow_other=True
#         )
#         for x in arrays
#     ]
#     if not any(
#         isinstance(
#             x,
#             (ak._v2.contents.Content, ak.partition.PartitionedArray, ak._v2.contents.Content),
#         )
#         for x in contents
#     ):
#         raise ValueError(
#             "need at least one array to concatenate"
#             + ak._v2._util.exception_suffix(__file__)
#         )

#     first_content = [
#         x
#         for x in contents
#         if isinstance(
#             x,
#             (ak._v2.contents.Content, ak.partition.PartitionedArray, ak._v2.contents.Content),
#         )
#     ][0]
#     posaxis = first_content.axis_wrap_if_negative(axis)
#     maxdepth = max(
#         [
#             x.minmax_depth[1]
#             for x in contents
#             if isinstance(
#                 x,
#                 (
#                     ak._v2.contents.Content,
#                     ak.partition.PartitionedArray,
#                     ak._v2.contents.Content,
#                 ),
#             )
#         ]
#     )
#     if not 0 <= posaxis < maxdepth:
#         raise ValueError(
#             "axis={0} is beyond the depth of this array or the depth of this array "
#             "is ambiguous".format(axis) + ak._v2._util.exception_suffix(__file__)
#         )
#     for x in contents:
#         if isinstance(x, ak._v2.contents.Content):
#             if x.axis_wrap_if_negative(axis) != posaxis:
#                 raise ValueError(
#                     "arrays to concatenate do not have the same depth for negative "
#                     "axis={0}".format(axis) + ak._v2._util.exception_suffix(__file__)
#                 )

#     if any(isinstance(x, ak.partition.PartitionedArray) for x in contents):
#         if posaxis == 0:
#             partitions = []
#             offsets = [0]
#             for content in contents:
#                 if isinstance(content, ak.partition.PartitionedArray):
#                     start = 0
#                     for stop, part in __builtins__["zip"](
#                         content.stops, content.partitions
#                     ):
#                         count = stop - start
#                         start = stop
#                         partitions.append(part)
#                         offsets.append(offsets[-1] + count)
#                 elif isinstance(content, ak._v2.contents.Content):
#                     partitions.append(content)
#                     offsets.append(offsets[-1] + len(content))
#                 else:
#                     partitions.append(
#                         ak.operations.convert.from_iter([content], highlevel=False)
#                     )
#                     offsets.append(offsets[-1] + 1)

#             out = ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])

#         else:
#             for content in contents:
#                 if isinstance(content, ak.partition.PartitionedArray):
#                     stops = content.stops
#                     slices = []
#                     start = 0
#                     for stop in stops:
#                         slices.append(slice(start, stop))
#                         start = stop
#                     break

#             partitions = []
#             offsets = [0]
#             for slc in slices:
#                 newcontents = []
#                 for content in contents:
#                     if isinstance(content, ak.partition.PartitionedArray):
#                         newcontents.append(content[slc].toContent())
#                     elif isinstance(content, ak._v2.contents.Content):
#                         newcontents.append(content[slc])
#                     else:
#                         newcontents.append(content)

#                 partitions.append(
#                     concatenate(
#                         newcontents,
#                         axis=axis,
#                         merge=merge,
#                         mergebool=mergebool,
#                         highlevel=False,
#                     )
#                 )
#                 offsets.append(offsets[-1] + len(partitions[-1]))

#             out = ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])

#     elif posaxis == 0:
#         contents = [
#             x
#             if isinstance(x, ak._v2.contents.Content)
#             else ak.operations.convert.to_layout([x])
#             for x in contents
#         ]
#         batch = [contents[0]]
#         for x in contents[1:]:
#             if batch[-1].mergeable(x, mergebool=mergebool):
#                 batch.append(x)
#             else:
#                 collapsed = batch[0].mergemany(batch[1:])
#                 batch = [collapsed.merge_as_union(x)]

#         out = batch[0].mergemany(batch[1:])
#         if isinstance(out, ak._v2._util.uniontypes):
#             out = out.simplify(merge=merge, mergebool=mergebool)

#     else:

#         def getfunction(inputs, depth):
#             if depth == posaxis and any(
#                 isinstance(x, ak._v2._util.optiontypes) for x in inputs
#             ):
#                 nextinputs = []
#                 for x in inputs:
#                     if isinstance(x, ak._v2._util.optiontypes) and isinstance(
#                         x.content, ak._v2._util.listtypes
#                     ):
#                         nextinputs.append(fill_none(x, [], axis=0, highlevel=False))
#                     else:
#                         nextinputs.append(x)
#                 inputs = nextinputs

#             if depth == posaxis and all(
#                 isinstance(x, ak._v2._util.listtypes)
#                 or (isinstance(x, ak._v2.contents.NumpyArray) and x.ndim > 1)
#                 or not isinstance(x, ak._v2.contents.Content)
#                 for x in inputs
#             ):
#                 nplike = ak.nplike.of(*inputs)

#                 length = max(
#                     [len(x) for x in inputs if isinstance(x, ak._v2.contents.Content)]
#                 )
#                 nextinputs = []
#                 for x in inputs:
#                     if isinstance(x, ak._v2.contents.Content):
#                         nextinputs.append(x)
#                     else:
#                         nextinputs.append(
#                             ak._v2.contents.ListOffsetArray64(
#                                 ak._v2.index.Index64(
#                                     nplike.arange(length + 1, dtype=np.int64)
#                                 ),
#                                 ak._v2.contents.NumpyArray(
#                                     nplike.broadcast_to(nplike.array([x]), (length,))
#                                 ),
#                             )
#                         )

#                 counts = nplike.zeros(len(nextinputs[0]), dtype=np.int64)
#                 all_counts = []
#                 all_flatten = []
#                 for x in nextinputs:
#                     o, f = x.offsets_and_flatten(1)
#                     o = nplike.asarray(o)
#                     c = o[1:] - o[:-1]
#                     nplike.add(counts, c, out=counts)
#                     all_counts.append(c)
#                     all_flatten.append(f)

#                 offsets = nplike.empty(len(nextinputs[0]) + 1, dtype=np.int64)
#                 offsets[0] = 0
#                 nplike.cumsum(counts, out=offsets[1:])

#                 offsets = ak._v2.index.Index64(offsets)
#                 tags, index = ak._v2.contents.UnionArray8_64.nested_tags_index(
#                     offsets,
#                     [ak._v2.index.Index64(x) for x in all_counts],
#                 )
#                 inner = ak._v2.contents.UnionArray8_64(tags, index, all_flatten)

#                 out = ak._v2.contents.ListOffsetArray64(
#                     offsets, inner.simplify(merge=merge, mergebool=mergebool)
#                 )
#                 return lambda: (out,)

#             elif any(
#                 x.minmax_depth == (1, 1)
#                 for x in inputs
#                 if isinstance(x, ak._v2.contents.Content)
#             ):
#                 raise ValueError(
#                     "at least one array is not deep enough to concatenate at "
#                     "axis={0}".format(axis) + ak._v2._util.exception_suffix(__file__)
#                 )

#             else:
#                 return None

#         out = ak._v2._util.broadcast_and_apply(
#             contents,
#             getfunction,
#             behavior=ak._v2._util.behaviorof(*arrays, behavior=behavior),
#             allow_records=True,
#             right_broadcast=False,
#             pass_depth=True,
#         )[0]

#     return ak._v2._util.maybe_wrap(
#         out, ak._v2._util.behaviorof(*arrays, behavior=behavior), highlevel
#     )
