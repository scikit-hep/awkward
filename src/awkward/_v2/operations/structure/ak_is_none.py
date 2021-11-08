# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def is_none(array, axis=0, highlevel=True, behavior=None):
    raise NotImplementedError


#     """
#     Args:
#         array: Data to check for missing values (None).
#         axis (int): The dimension at which this operation is applied. The
#             outermost dimension is `0`, followed by `1`, etc., and negative
#             values count backward from the innermost: `-1` is the innermost
#             dimension, `-2` is the next level up, etc.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Returns an array whose value is True where an element of `array` is None;
#     False otherwise (at a given `axis` depth).
#     """

#     def getfunction(layout, depth, posaxis):
#         posaxis = layout.axis_wrap_if_negative(posaxis)
#         if posaxis == depth - 1:
#             nplike = ak.nplike.of(layout)
#             if isinstance(layout, ak._v2._util.optiontypes):
#                 return lambda: ak._v2.contents.NumpyArray(
#                     nplike.asarray(layout.bytemask()).view(np.bool_)
#                 )
#             elif isinstance(
#                 layout,
#                 (
#                     ak._v2._util.unknowntypes,
#                     ak._v2._util.listtypes,
#                     ak._v2._util.recordtypes,
#                     ak._v2.contents.NumpyArray,
#                 ),
#             ):
#                 return lambda: ak._v2.contents.NumpyArray(
#                     nplike.zeros(len(layout), dtype=np.bool_)
#                 )
#             else:
#                 return posaxis
#         else:
#             return posaxis

#     layout = ak._v2.operations.convert.to_layout(array)

#     out = ak._v2._util.recursively_apply(
#         layout, getfunction, pass_depth=True, pass_user=True, user=axis
#     )

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
