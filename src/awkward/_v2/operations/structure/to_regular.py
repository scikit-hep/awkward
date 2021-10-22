# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_regular(array, axis=1, highlevel=True, behavior=None):
    pass


#     """
#     Args:
#         array: Array to convert.
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Converts a variable-length axis into a regular one, if possible.

#         >>> irregular = ak.from_iter(np.arange(2*3*5).reshape(2, 3, 5))
#         >>> ak.type(irregular)
#         2 * var * var * int64
#         >>> ak.type(ak.to_regular(irregular))
#         2 * 3 * var * int64
#         >>> ak.type(ak.to_regular(irregular, axis=2))
#         2 * var * 5 * int64
#         >>> ak.type(ak.to_regular(irregular, axis=-1))
#         2 * var * 5 * int64

#     But truly irregular data cannot be converted.

#         >>> ak.to_regular(ak.Array([[1, 2, 3], [], [4, 5]]))
#         ValueError: in ListOffsetArray64, cannot convert to RegularArray because
#         subarray lengths are not regular

#     See also #ak.from_regular.
#     """

#     def getfunction(layout, depth, posaxis):
#         posaxis = layout.axis_wrap_if_negative(posaxis)
#         if posaxis == depth and isinstance(layout, ak._v2.contents.RegularArray):
#             return lambda: layout
#         elif posaxis == depth and isinstance(layout, ak._v2._util.listtypes):
#             return lambda: layout.toRegularArray()
#         elif posaxis == 0:
#             raise ValueError(
#                 "array has no axis {0}".format(axis)
#
#             )
#         else:
#             return posaxis

#     out = ak._v2.operations.convert.to_layout(array)
#     if axis != 0:
#         out = ak._v2._util.recursively_apply(
#             out,
#             getfunction,
#             pass_depth=True,
#             pass_user=True,
#             user=axis,
#             numpy_to_regular=True,
#         )

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
