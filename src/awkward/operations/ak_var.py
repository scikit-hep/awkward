# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("var",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import maybe_posaxis
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import regularize_axis

np = NumpyMetadata.instance()


@high_level_function()
def var(x, weight=None, ddof=0, axis=None, *, keepdims=False, mask_identity=False):
    """
    Args:
        x: The data on which to compute the variance (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            variance."
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

    Computes the variance in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity. It is the same as NumPy's
    [var](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the variance is calculated as

        ak.sum((x - ak.mean(x))**2 * weight) / ak.sum(weight)

    If `ddof` is not zero, the above is further corrected by a factor of

        ak.sum(weight) / (ak.sum(weight) - ddof)

    Even without `ddof`, #ak.var differs from #ak.moment with `n=2` because
    the mean is subtracted from all points before summing their squares.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.

    See also #ak.nanvar.
    """
    # Dispatch
    yield x, weight

    # Implementation
    return _impl(x, weight, ddof, axis, keepdims, mask_identity)


@high_level_function()
def nanvar(x, weight=None, ddof=0, axis=None, *, keepdims=False, mask_identity=True):
    """
    Args:
        x: The data on which to compute the variance (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            variance."
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

    Like #ak.var, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.var(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.var.
    """
    # Dispatch
    yield x, weight

    # Implementation
    if weight is not None:
        weight = ak.operations.ak_nan_to_none._impl(weight, False, None)

    return _impl(
        ak.operations.ak_nan_to_none._impl(x, False, None),
        weight,
        ddof,
        axis,
        keepdims,
        mask_identity,
    )


def _impl(x, weight, ddof, axis, keepdims, mask_identity):
    axis = regularize_axis(axis)
    behavior = behavior_of(x, weight)
    x = ak.highlevel.Array(
        ak.operations.to_layout(x, allow_record=False, allow_other=False),
        behavior=behavior,
    )
    if weight is not None:
        weight = ak.highlevel.Array(
            ak.operations.to_layout(weight, allow_record=False, allow_other=False),
            behavior=behavior,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        xmean = ak.operations.ak_mean._impl(
            x, weight, axis, keepdims=True, mask_identity=True
        )
        if weight is None:
            sumw = ak.operations.ak_count._impl(
                x,
                axis,
                keepdims=True,
                mask_identity=True,
                highlevel=True,
                behavior=None,
            )
            sumwxx = ak.operations.ak_sum._impl(
                (x - xmean) ** 2,
                axis,
                keepdims=True,
                mask_identity=True,
                highlevel=True,
                behavior=None,
            )
        else:
            sumw = ak.operations.ak_sum._impl(
                x * 0 + weight,
                axis,
                keepdims=True,
                mask_identity=True,
                highlevel=True,
                behavior=None,
            )
            sumwxx = ak.operations.ak_sum._impl(
                (x - xmean) ** 2 * weight,
                axis,
                keepdims=True,
                mask_identity=True,
                highlevel=True,
                behavior=None,
            )
        if ddof != 0:
            out = (sumwxx / sumw) * (sumw / (sumw - ddof))
        else:
            out = sumwxx / sumw

        if not mask_identity:
            out = ak.highlevel.Array(ak.operations.fill_none(out, np.nan, axis=-1))

        if axis is None:
            if not keepdims:
                out = out[(0,) * out.ndim]
        else:
            if not keepdims:
                posaxis = maybe_posaxis(out.layout, axis, 1)
                out = out[(slice(None, None),) * posaxis + (0,)]

        return out


@ak._connect.numpy.implements("var")
def _nep_18_impl_var(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    ddof=0,
    keepdims=False,
    *,
    where=UNSUPPORTED,
):
    return var(a, axis=axis, keepdims=keepdims, ddof=ddof)


@ak._connect.numpy.implements("nanvar")
def _nep_18_impl_nanvar(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    ddof=0,
    keepdims=False,
    *,
    where=UNSUPPORTED,
):
    return nanvar(a, axis=axis, keepdims=keepdims, ddof=ddof)
