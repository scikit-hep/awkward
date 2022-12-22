# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._util import unset

np = ak._nplikes.NumpyMetadata.instance()


def covar(
    x,
    y,
    weight=None,
    axis=None,
    *,
    keepdims=False,
    mask_identity=False,
    flatten_records=unset,
):
    """
    Args:
        x: One coordinate to use in the covariance calculation (anything #ak.to_layout recognizes).
        y: The other coordinate to use in the covariance calculation (anything #ak.to_layout recognizes).
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

    Computes the covariance of `x` and `y` (many types supported, including
    all Awkward Arrays and Records, must be broadcastable to each other).
    The grouping is performed the same way as for reducers, though this
    operation is not a reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the covariance is calculated as

        ak.sum((x - ak.mean(x))*(y - ak.mean(y))*weight) / ak.sum(weight)

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with ak._errors.OperationErrorContext(
        "ak.covar",
        dict(
            x=x,
            y=y,
            weight=weight,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
        ),
    ):
        if flatten_records is not unset:
            raise ak._errors.wrap_error(
                ValueError(
                    "`flatten_records` is no longer a supported argument for reducers. "
                    "Instead, use `ak.ravel(array)` first to remove the record structure "
                    "and flatten the array."
                )
            )
        return _impl(x, y, weight, axis, keepdims, mask_identity)


def _impl(x, y, weight, axis, keepdims, mask_identity):
    behavior = ak._util.behavior_of(x, y, weight)
    x = ak.highlevel.Array(
        ak.operations.to_layout(x, allow_record=False, allow_other=False),
        behavior=behavior,
    )
    y = ak.highlevel.Array(
        ak.operations.to_layout(y, allow_record=False, allow_other=False),
        behavior=behavior,
    )
    if weight is not None:
        weight = ak.highlevel.Array(
            ak.operations.to_layout(weight, allow_record=False, allow_other=False),
            behavior=behavior,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        xmean = ak.operations.ak_mean._impl(x, weight, axis, False, mask_identity)
        ymean = ak.operations.ak_mean._impl(y, weight, axis, False, mask_identity)
        if weight is None:
            sumw = ak.operations.ak_count._impl(
                x,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=behavior,
            )
            sumwxy = ak.operations.ak_sum._impl(
                (x - xmean) * (y - ymean),
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=behavior,
            )
        else:
            sumw = ak.operations.ak_sum._impl(
                x * 0 + weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=behavior,
            )
            sumwxy = ak.operations.ak_sum._impl(
                (x - xmean) * (y - ymean) * weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=behavior,
            )
        return ak._nplikes.nplike_of(sumwxy, sumw).true_divide(sumwxy, sumw)
