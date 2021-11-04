# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def firsts(array, axis=1, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data from which to select the first elements from nested lists.
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Selects the first element of each non-empty list and inserts None for each
#     empty list.

#     For example,

#         >>> array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]])
#         >>> print(ak.firsts(array))
#         [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]

#     See #ak.singletons to invert this function.
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     if isinstance(layout, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#         posaxis = None
#         for x in layout.partitions:
#             if posaxis is None:
#                 posaxis = x.axis_wrap_if_negative(axis)
#             elif posaxis != x.axis_wrap_if_negative(axis):
#                 raise ValueError(
#                     "ak.firsts for partitions with different axis depths"
#
#                 )
#     else:
#         posaxis = layout.axis_wrap_if_negative(axis)

#     if posaxis == 0:
#         if len(layout) == 0:
#             out = None
#         else:
#             out = layout[0]
#     else:
#         if posaxis < 0:
#             raise NotImplementedError(
#                 "ak.firsts with ambiguous negative axis"
#
#             )
#         toslice = (slice(None, None, None),) * posaxis + (0,)
#         out = ak.mask(layout, ak.num(layout, axis=posaxis) > 0, highlevel=False)[
#             toslice
#         ]

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
