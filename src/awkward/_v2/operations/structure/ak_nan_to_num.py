# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("nan_to_num")
def nan_to_num(
    array, copy=True, nan=0.0, posinf=None, neginf=None, highlevel=True, behavior=None
):
    raise NotImplementedError


#     """
#     Args:
#         array: Array whose `NaN` values should be converted to a number.
#         copy (bool): Ignored (Awkward Arrays are immutable).
#         nan (int or float): Value to be used to fill `NaN` values.
#         posinf (int, float, or None): Value to be used to fill positive infinity
#             values. If None, positive infinities are replaced with a very large number.
#         neginf (int, float, or None): Value to be used to fill negative infinity
#             values. If None, negative infinities are replaced with a very small number.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Implements [np.nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
#     for Awkward Arrays.
#     """
#     layout = ak._v2.operations.convert.to_layout(array)
#     nplike = ak.nplike.of(layout)

#     def getfunction(layout):
#         if isinstance(layout, ak._v2.contents.NumpyArray):
#             return lambda: ak._v2.contents.NumpyArray(
#                 nplike.nan_to_num(
#                     nplike.asarray(layout),
#                     nan=nan,
#                     posinf=posinf,
#                     neginf=neginf,
#                 )
#             )
#         else:
#             return None

#     out = ak._v2._util.recursively_apply(
#         layout, getfunction, pass_depth=False, pass_user=False
#     )
#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
