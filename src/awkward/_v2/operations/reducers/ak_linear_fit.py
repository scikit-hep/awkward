# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def linear_fit(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise NotImplementedError


#     """
#     Args:
#         x: One coordinate to use in the linear fit (anything #ak.to_layout recognizes).
#         y: The other coordinate to use in the linear fit (anything #ak.to_layout recognizes).
#         weight: Data that can be broadcasted to `x` and `y` to give each point
#             a weight. Weighting points equally is the same as no weights;
#             weighting some points higher increases the significance of those
#             points. Weights can be zero or negative.
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

#     Computes the linear fit of `y` with respect to `x` (many types supported,
#     including all Awkward Arrays and Records, must be broadcastable to each
#     other). The grouping is performed the same way as for reducers, though
#     this operation is not a reducer and has no identity.

#     This function has no NumPy equivalent.

#     Passing all arguments to the reducers, the linear fit is calculated as

#         sumw            = ak.sum(weight)
#         sumwx           = ak.sum(weight * x)
#         sumwy           = ak.sum(weight * y)
#         sumwxx          = ak.sum(weight * x**2)
#         sumwxy          = ak.sum(weight * x * y)
#         delta           = (sumw*sumwxx) - (sumwx*sumwx)

#         intercept       = ((sumwxx*sumwy) - (sumwx*sumwxy)) / delta
#         slope           = ((sumw*sumwxy) - (sumwx*sumwy))   / delta
#         intercept_error = np.sqrt(sumwxx / delta)
#         slope_error     = np.sqrt(sumw   / delta)

#     The results, `intercept`, `slope`, `intercept_error`, and `slope_error`,
#     are given as an #ak.Record with four fields. The values of these fields
#     might be arrays or even nested arrays; they match the structure of `x` and
#     `y`.

#     See #ak.sum for a complete description of handling nested lists and
#     missing values (None) in reducers, and #ak.mean for an example with another
#     non-reducer.
#     """
#     with np.errstate(invalid="ignore"):
#         nplike = ak.nplike.of(x, y, weight)
#         if weight is None:
#             sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
#             sumwx = sum(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
#             sumwy = sum(y, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
#             sumwxx = sum(
#                 x ** 2, axis=axis, keepdims=keepdims, mask_identity=mask_identity
#             )
#             sumwxy = sum(
#                 x * y, axis=axis, keepdims=keepdims, mask_identity=mask_identity
#             )
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
#             sumwy = sum(
#                 y * weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
#             )
#             sumwxx = sum(
#                 (x ** 2) * weight,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#             sumwxy = sum(
#                 x * y * weight,
#                 axis=axis,
#                 keepdims=keepdims,
#                 mask_identity=mask_identity,
#             )
#         delta = (sumw * sumwxx) - (sumwx * sumwx)
#         intercept = nplike.true_divide(((sumwxx * sumwy) - (sumwx * sumwxy)), delta)
#         slope = nplike.true_divide(((sumw * sumwxy) - (sumwx * sumwy)), delta)
#         intercept_error = nplike.sqrt(nplike.true_divide(sumwxx, delta))
#         slope_error = nplike.sqrt(nplike.true_divide(sumw, delta))

#         intercept = ak._v2.operations.convert.to_layout(
#             intercept, allow_record=True, allow_other=True
#         )
#         slope = ak._v2.operations.convert.to_layout(
#             slope, allow_record=True, allow_other=True
#         )
#         intercept_error = ak._v2.operations.convert.to_layout(
#             intercept_error, allow_record=True, allow_other=True
#         )
#         slope_error = ak._v2.operations.convert.to_layout(
#             slope_error, allow_record=True, allow_other=True
#         )

#         scalar = False
#         if not isinstance(
#             intercept,
#             (
#                 ak._v2.contents.Content,
#                 ak._v2.record.Record,
#                 ak.partition.PartitionedArray,   # NO PARTITIONED ARRAY
#             ),
#         ):
#             intercept = ak._v2.contents.NumpyArray(nplike.array([intercept]))
#             scalar = True
#         if not isinstance(
#             slope,
#             (
#                 ak._v2.contents.Content,
#                 ak._v2.record.Record,
#                 ak.partition.PartitionedArray,   # NO PARTITIONED ARRAY
#             ),
#         ):
#             slope = ak._v2.contents.NumpyArray(nplike.array([slope]))
#             scalar = True
#         if not isinstance(
#             intercept_error,
#             (
#                 ak._v2.contents.Content,
#                 ak._v2.record.Record,
#                 ak.partition.PartitionedArray,   # NO PARTITIONED ARRAY
#             ),
#         ):
#             intercept_error = ak._v2.contents.NumpyArray(nplike.array([intercept_error]))
#             scalar = True
#         if not isinstance(
#             slope_error,
#             (
#                 ak._v2.contents.Content,
#                 ak._v2.record.Record,
#                 ak.partition.PartitionedArray,   # NO PARTITIONED ARRAY
#             ),
#         ):
#             slope_error = ak._v2.contents.NumpyArray(nplike.array([slope_error]))
#             scalar = True

#         sample = None
#         if isinstance(intercept, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#             sample = intercept
#         elif isinstance(slope, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#             sample = slope
#         elif isinstance(intercept_error, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#             sample = intercept_error
#         elif isinstance(slope_error, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
#             sample = slope_error

#         if sample is not None:
#             (
#                 intercept,
#                 slope,
#                 intercept_error,
#                 slope_error,
#             ) = ak.partition.partition_as(
#                 sample, (intercept, slope, intercept_error, slope_error)
#             )
#             output = []
#             for a, b, c, d in ak.partition.iterate(
#                 sample.numpartitions, (intercept, slope, intercept_error, slope_error)
#             ):
#                 output.append(
#                     ak._v2.contents.RecordArray(
#                         [a, b, c, d],
#                         ["intercept", "slope", "intercept_error", "slope_error"],
#                         parameters={"__record__": "LinearFit"},
#                     )
#                 )
#             out = ak.partition.IrregularlyPartitionedArray(output)   # NO PARTITIONED ARRAY

#         else:
#             out = ak._v2.contents.RecordArray(
#                 [intercept, slope, intercept_error, slope_error],
#                 ["intercept", "slope", "intercept_error", "slope_error"],
#                 parameters={"__record__": "LinearFit"},
#             )
#             if scalar:
#                 out = out[0]

#         return ak._v2._util.wrap(out, ak._v2._util.behaviorof(x, y))
