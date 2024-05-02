# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes import ufuncs
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("linear_fit",)

np = NumpyMetadata.instance()


@high_level_function()
def linear_fit(
    x,
    y,
    weight=None,
    axis=None,
    *,
    keepdims=False,
    mask_identity=False,
    highlevel=True,
    behavior=None,
    attrs=None,
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
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

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
    # Dispatch
    yield x, y, weight

    # Implementation
    return _impl(
        x, y, weight, axis, keepdims, mask_identity, highlevel, behavior, attrs
    )


def _impl(x, y, weight, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    axis = regularize_axis(axis)

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        x_layout, y_layout, weight_layout = ensure_same_backend(
            ctx.unwrap(x, allow_record=False, primitive_policy="error"),
            ctx.unwrap(y, allow_record=False, primitive_policy="error"),
            ctx.unwrap(
                weight,
                allow_record=False,
                allow_unknown=False,
                primitive_policy="error",
                none_policy="pass-through",
            ),
        )

    x = ctx.wrap(x_layout)
    y = ctx.wrap(y_layout)
    weight = ctx.wrap(weight_layout, allow_other=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        if weight is None:
            sumw = ak.operations.ak_count._impl(
                x,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwx = ak.operations.ak_sum._impl(
                x,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwy = ak.operations.ak_sum._impl(
                y,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwxx = ak.operations.ak_sum._impl(
                x**2,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwxy = ak.operations.ak_sum._impl(
                x * y,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
        else:
            sumw = ak.operations.ak_sum._impl(
                x * 0 + weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwx = ak.operations.ak_sum._impl(
                x * weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwy = ak.operations.ak_sum._impl(
                y * weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwxx = ak.operations.ak_sum._impl(
                (x**2) * weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwxy = ak.operations.ak_sum._impl(
                x * y * weight,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
        delta = (sumw * sumwxx) - (sumwx * sumwx)
        intercept = ((sumwxx * sumwy) - (sumwx * sumwxy)) / delta
        slope = ((sumw * sumwxy) - (sumwx * sumwy)) / delta
        intercept_error = ufuncs.sqrt(sumwxx / delta)
        slope_error = ufuncs.sqrt(sumw / delta)

        is_scalar = not isinstance(
            ak.operations.to_layout(intercept, primitive_policy="pass-through"),
            ak.contents.Content,
        )

        intercept = ak.operations.to_layout(intercept)
        slope = ak.operations.to_layout(slope)
        intercept_error = ak.operations.to_layout(intercept_error)
        slope_error = ak.operations.to_layout(slope_error)

        out = ak.contents.RecordArray(
            [intercept, slope, intercept_error, slope_error],
            ["intercept", "slope", "intercept_error", "slope_error"],
            parameters={"__record__": "LinearFit"},
        )
        if is_scalar:
            out = out[0]

        return ctx.wrap(out, highlevel=highlevel, allow_other=is_scalar)
