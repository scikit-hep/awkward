# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("sort")
def sort(array, axis=-1, ascending=True, stable=True, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data to sort, possibly within nested lists.
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         ascending (bool): If True, the first value in each sorted group
#             will be smallest, the last value largest; if False, the order
#             is from largest to smallest.
#         stable (bool): If True, use a stable sorting algorithm (introsort:
#             a hybrid of quicksort, heapsort, and insertion sort); if False,
#             use a sorting algorithm that is not guaranteed to be stable
#             (heapsort).
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     For example,

#         >>> ak.sort(ak.Array([[7, 5, 7], [], [2], [8, 2]]))
#         <Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
#     """
#     layout = ak._v2.operations.convert.to_layout(
#         array, allow_record=False, allow_other=False
#     )
#     out = layout.sort(axis, ascending, stable)
#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
