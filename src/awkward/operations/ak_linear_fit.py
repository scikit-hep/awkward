# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
    """
    Args:
        x: One coordinate to use in the linear fit (anything #ak.to_layout recognizes).
        y: The other coordinate to use in the linear fit (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` and `y` to give each point
            a weight. Weighting points equally is the same as no weights;
            weighting some points higher increases the significance of those
            points. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.
        flatten_records (bool): If True, axis=None combines fields from different
            records; otherwise, records raise an error.

    Computes the linear fit of `y` with respect to `x` (many types supported,
    including all Awkward Arrays and Records, must be broadcastable to each
    other). The grouping is performed the same way as for reducers, though
    this operation is not a reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the linear fit is calculated as

        sumw            = ak.sum(weight)
        sumwx           = ak.sum(weight * x)
        sumwy           = ak.sum(weight * y)
        sumwxx          = ak.sum(weight * x**2)
        sumwxy          = ak.sum(weight * x * y)
        delta           = (sumw*sumwxx) - (sumwx*sumwx)

        intercept       = ((sumwxx*sumwy) - (sumwx*sumwxy)) / delta
        slope           = ((sumw*sumwxy) - (sumwx*sumwy))   / delta
        intercept_error = np.sqrt(sumwxx / delta)
        slope_error     = np.sqrt(sumw   / delta)

    The results, `intercept`, `slope`, `intercept_error`, and `slope_error`,
    are given as an #ak.Record with four fields. The values of these fields
    might be arrays or even nested arrays; they match the structure of `x` and
    `y`.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.linear_fit",
        dict(
            x=x,
            y=y,
            weight=weight,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        ),
    ):
        return _impl(x, y, weight, axis, keepdims, mask_identity, flatten_records)


def _impl(x, y, weight, axis, keepdims, mask_identity, flatten_records):
    x = ak._v2.highlevel.Array(
        ak._v2.operations.to_layout(x, allow_record=False, allow_other=False)
    )
    y = ak._v2.highlevel.Array(
        ak._v2.operations.to_layout(y, allow_record=False, allow_other=False)
    )
    if weight is not None:
        weight = ak._v2.highlevel.Array(
            ak._v2.operations.to_layout(weight, allow_record=False, allow_other=False)
        )

    with np.errstate(invalid="ignore"):
        nplike = ak.nplike.of(x, y, weight)
        if weight is None:
            sumw = ak._v2.operations.ak_count._impl(
                x, axis, keepdims, mask_identity, flatten_records
            )
            sumwx = ak._v2.operations.ak_sum._impl(
                x, axis, keepdims, mask_identity, flatten_records
            )
            sumwy = ak._v2.operations.ak_sum._impl(
                y, axis, keepdims, mask_identity, flatten_records
            )
            sumwxx = ak._v2.operations.ak_sum._impl(
                x**2, axis, keepdims, mask_identity, flatten_records
            )
            sumwxy = ak._v2.operations.ak_sum._impl(
                x * y, axis, keepdims, mask_identity, flatten_records
            )
        else:
            sumw = ak._v2.operations.ak_sum._impl(
                x * 0 + weight,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
            )
            sumwx = ak._v2.operations.ak_sum._impl(
                x * weight, axis, keepdims, mask_identity, flatten_records
            )
            sumwy = ak._v2.operations.ak_sum._impl(
                y * weight, axis, keepdims, mask_identity, flatten_records
            )
            sumwxx = ak._v2.operations.ak_sum._impl(
                (x**2) * weight,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
            )
            sumwxy = ak._v2.operations.ak_sum._impl(
                x * y * weight,
                axis,
                keepdims,
                mask_identity,
                flatten_records,
            )
        delta = (sumw * sumwxx) - (sumwx * sumwx)
        intercept = nplike.true_divide(((sumwxx * sumwy) - (sumwx * sumwxy)), delta)
        slope = nplike.true_divide(((sumw * sumwxy) - (sumwx * sumwy)), delta)
        intercept_error = nplike.sqrt(nplike.true_divide(sumwxx, delta))
        slope_error = nplike.sqrt(nplike.true_divide(sumw, delta))

        intercept = ak._v2.operations.to_layout(
            intercept, allow_record=True, allow_other=True
        )
        slope = ak._v2.operations.to_layout(slope, allow_record=True, allow_other=True)
        intercept_error = ak._v2.operations.to_layout(
            intercept_error, allow_record=True, allow_other=True
        )
        slope_error = ak._v2.operations.to_layout(
            slope_error, allow_record=True, allow_other=True
        )

        scalar = False
        if not isinstance(
            intercept,
            (
                ak._v2.contents.Content,
                ak._v2.record.Record,
            ),
        ):
            intercept = ak._v2.contents.NumpyArray(nplike.array([intercept]))
            scalar = True
        if not isinstance(
            slope,
            (
                ak._v2.contents.Content,
                ak._v2.record.Record,
            ),
        ):
            slope = ak._v2.contents.NumpyArray(nplike.array([slope]))
            scalar = True
        if not isinstance(
            intercept_error,
            (
                ak._v2.contents.Content,
                ak._v2.record.Record,
            ),
        ):
            intercept_error = ak._v2.contents.NumpyArray(
                nplike.array([intercept_error])
            )
            scalar = True
        if not isinstance(
            slope_error,
            (
                ak._v2.contents.Content,
                ak._v2.record.Record,
            ),
        ):
            slope_error = ak._v2.contents.NumpyArray(nplike.array([slope_error]))
            scalar = True

        out = ak._v2.contents.RecordArray(
            [intercept, slope, intercept_error, slope_error],
            ["intercept", "slope", "intercept_error", "slope_error"],
            parameters={"__record__": "LinearFit"},
        )
        if scalar:
            out = out[0]

        if isinstance(out, (ak._v2.contents.Content, ak._v2.record.Record)):
            return ak._v2._util.wrap(out, ak._v2._util.behavior_of(x, y))
        else:
            return out
