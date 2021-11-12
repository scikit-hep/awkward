# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("mean")
def mean(
    x, weight=None, axis=None, keepdims=False, mask_identity=True, flatten_records=False
):
    raise NotImplementedError


#     """
#     Args:
#         x: The data on which to compute the mean (anything #ak.to_layout recognizes).
#         weight: Data that can be broadcasted to `x` to give each value a
#             weight. Weighting values equally is the same as no weights;
#             weighting some values higher increases the significance of those
#             values. Weights can be zero or negative.
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

#     Computes the mean in each group of elements from `x` (many
#     types supported, including all Awkward Arrays and Records). The grouping
#     is performed the same way as for reducers, though this operation is not a
#     reducer and has no identity. It is the same as NumPy's
#     [mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)
#     if all lists at a given dimension have the same length and no None values,
#     but it generalizes to cases where they do not.

#     Passing all arguments to the reducers, the mean is calculated as

#         ak.sum(x*weight) / ak.sum(weight)

#     For example, with an `array` like

#         ak.Array([[0, 1, 2, 3],
#                   [          ],
#                   [4, 5      ]])

#     The mean of the innermost lists is

#         >>> ak.mean(array, axis=-1)
#         <Array [1.5, None, 4.5] type='3 * ?float64'>

#     because there are three lists, the first has mean `1.5`, the second is
#     empty, and the third has mean `4.5`.

#     The mean of the outermost lists is

#         >>> ak.mean(array, axis=0)
#         <Array [2, 3, 2, 3] type='4 * ?float64'>

#     because the longest list has length 4, the mean of `0` and `4` is `2.0`,
#     the mean of `1` and `5` is `3.0`, the mean of `2` (by itself) is `2.0`,
#     and the mean of `3` (by itself) is `3.0`. This follows the same grouping
#     behavior as reducers.

#     See #ak.sum for a complete description of handling nested lists and
#     missing values (None) in reducers.
#     """
#     with np.errstate(invalid="ignore"):
#         if weight is None:
#             sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
#             sumwx = sum(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
#         else:
#             sumw = sum(
#                 x * 0 + weight,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#             sumwx = sum(
#                 x * weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
#             )
#         return ak.nplike.of(sumwx, sumw).true_divide(sumwx, sumw)
