# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("std")
def std(
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
#         x: The data on which to compute the standard deviation (anything #ak.to_layout recognizes).
#         weight: Data that can be broadcasted to `x` to give each value a
#             weight. Weighting values equally is the same as no weights;
#             weighting some values higher increases the significance of those
#             values. Weights can be zero or negative.
#         ddof (int): "delta degrees of freedom": the divisor used in the
#             calculation is `sum(weights) - ddof`. Use this for "reduced
#             standard deviation."
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

#     Computes the standard deviation in each group of elements from `x`
#     (many types supported, including all Awkward Arrays and Records). The
#     grouping is performed the same way as for reducers, though this operation
#     is not a reducer and has no identity. It is the same as NumPy's
#     [std](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)
#     if all lists at a given dimension have the same length and no None values,
#     but it generalizes to cases where they do not.

#     Passing all arguments to the reducers, the standard deviation is
#     calculated as

#         np.sqrt(ak.var(x, weight))

#     See #ak.sum for a complete description of handling nested lists and
#     missing values (None) in reducers, and #ak.mean for an example with another
#     non-reducer.
#     """
#     with np.errstate(invalid="ignore"):
#         return ak.nplike.of(x, weight).sqrt(
#             var(
#                 x,
#                 weight=weight,
#                 ddof=ddof,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#         )
