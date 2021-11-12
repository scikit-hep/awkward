# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("var")
def var(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise NotImplementedError


#     """
#     Args:
#         x: The data on which to compute the variance (anything #ak.to_layout recognizes).
#         weight: Data that can be broadcasted to `x` to give each value a
#             weight. Weighting values equally is the same as no weights;
#             weighting some values higher increases the significance of those
#             values. Weights can be zero or negative.
#         ddof (int): "delta degrees of freedom": the divisor used in the
#             calculation is `sum(weights) - ddof`. Use this for "reduced
#             variance."
#         axis (None or int): If None, combine all values from the array into
#             a single scalar result; if an int, group by that axis: `0` is the
#             outermost, `1` is the first level of nested lists, etc., and
#             negative `axis` counts from the innermost: `-1` is the innermost,
#             `-2` is the next level up, etc.
#         keepdims (bool): If False, this function decreases the number of
#             dimensions by 1; if True, the output values are wrapped in a new
#             length-1 dimension so that the result of this operation may be
#             broadcasted with the original array.
#         mask_identity (bool): If True, the application of this function on
#             empty lists results in None (an option type); otherwise, the
#             calculation is followed through with the reducers' identities,
#             usually resulting in floating-point `nan`.
#         flatten_records (bool): If True, axis=None combines fields from different
#             records; otherwise, records raise an error.

#     Computes the variance in each group of elements from `x` (many
#     types supported, including all Awkward Arrays and Records). The grouping
#     is performed the same way as for reducers, though this operation is not a
#     reducer and has no identity. It is the same as NumPy's
#     [var](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)
#     if all lists at a given dimension have the same length and no None values,
#     but it generalizes to cases where they do not.

#     Passing all arguments to the reducers, the variance is calculated as

#         ak.sum((x - ak.mean(x))**2 * weight) / ak.sum(weight)

#     If `ddof` is not zero, the above is further corrected by a factor of

#         ak.sum(weight) / (ak.sum(weight) - ddof)

#     Even without `ddof`, #ak.var differs from #ak.moment with `n=2` because
#     the mean is subtracted from all points before summing their squares.

#     See #ak.sum for a complete description of handling nested lists and
#     missing values (None) in reducers, and #ak.mean for an example with another
#     non-reducer.
#     """
#     with np.errstate(invalid="ignore"):
#         xmean = mean(
#             x, weight=weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
#         )
#         if weight is None:
#             sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
#             sumwxx = sum(
#                 (x - xmean) ** 2,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#         else:
#             sumw = sum(
#                 x * 0 + weight,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#             sumwxx = sum(
#                 (x - xmean) ** 2 * weight,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#         if ddof != 0:
#             return ak.nplike.of(sumwxx, sumw).true_divide(sumwxx, sumw) * ak.nplike.of(
#                 sumw
#             ).true_divide(sumw, sumw - ddof)
#         else:
#             return ak.nplike.of(sumwxx, sumw).true_divide(sumwxx, sumw)
