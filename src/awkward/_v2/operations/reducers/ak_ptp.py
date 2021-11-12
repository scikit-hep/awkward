# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("ptp")
def ptp(arr, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    raise NotImplementedError


#     """
#     Args:
#         array: Array-like data (anything #ak.to_layout recognizes).
#         axis (None or int): If None, combine all values from the array into
#             a single scalar result; if an int, group by that axis: `0` is the
#             outermost, `1` is the first level of nested lists, etc., and
#             negative `axis` counts from the innermost: `-1` is the innermost,
#             `-2` is the next level up, etc.
#         keepdims (bool): If False, this reducer decreases the number of
#             dimensions by 1; if True, the reduced values are wrapped in a new
#             length-1 dimension so that the result of this operation may be
#             broadcasted with the original array.
#         mask_identity (bool): If True, reducing over empty lists results in
#             None (an option type); otherwise, reducing over empty lists
#             results in the operation's identity of 0.
#         flatten_records (bool): If True, axis=None combines fields from different
#             records; otherwise, records raise an error.

#     Returns the range of values in each group of elements from `array` (many
#     types supported, including all Awkward Arrays and Records). The range of
#     an empty list is None, unless `mask_identity=False`, in which case it is 0.
#     This operation is the same as NumPy's
#     [ptp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html)
#     if all lists at a given dimension have the same length and no None values,
#     but it generalizes to cases where they do not.

#     For example, with an `array` like

#         ak.Array([[0, 1, 2, 3],
#                   [          ],
#                   [4, 5      ]])

#     The range of the innermost lists is

#         >>> ak.ptp(array, axis=-1)
#         <Array [3, None, 1] type='3 * ?float64'>

#     because there are three lists, the first has a range of `3`, the second is
#     `None` because the list is empty, and the third has a range of `1`. Similarly,

#         >>> ak.ptp(array, axis=-1, mask_identity=False)
#         <Array [3, 0, 1] type='3 * float64'>

#     The second value is `0` because the list is empty.

#     See #ak.sum for a more complete description of nested list and missing
#     value (None) handling in reducers.
#     """
#     if axis is None:
#         out = ak.max(arr) - ak.min(arr)
#         if not mask_identity and out is None:
#             out = 0

#     else:
#         maxi = ak.max(arr, axis=axis, mask_identity=True, keepdims=True)
#         mini = ak.min(arr, axis=axis, mask_identity=True, keepdims=True)
#         out = maxi - mini

#         if not mask_identity:
#             out = ak.fill_none(out, 0, axis=-1)

#         if not keepdims:
#             posaxis = out.layout.axis_wrap_if_negative(axis)
#             out = out[(slice(None, None),) * posaxis + (0,)]

#     return out
