# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import (
    HighLevelContext,
    ensure_same_backend,
    maybe_highlevel_to_lowlevel,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("moment",)

np = NumpyMetadata.instance()


@high_level_function()
def moment(
    x,
    n,
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
        x: The data on which to compute the moment (anything #ak.to_layout recognizes).
        n (int): The choice of moment: `0` is a sum of weights, `1` is
            #ak.mean, `2` is #ak.var without subtracting the mean, etc.
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
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

    Computes the `n`th moment in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the moment is calculated as

        ak.sum((x*weight)**n) / ak.sum(weight)

    The `n=2` moment differs from #ak.var in that #ak.var also subtracts the
    mean (the `n=1` moment).

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    # Dispatch
    yield x, weight

    # Implementation
    return _impl(
        x, n, weight, axis, keepdims, mask_identity, highlevel, behavior, attrs
    )


def _impl(x, n, weight, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    axis = regularize_axis(axis)

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        x_layout, weight_layout = ensure_same_backend(
            ctx.unwrap(x, allow_record=False, primitive_policy="error"),
            ctx.unwrap(
                weight,
                allow_record=False,
                allow_unknown=False,
                primitive_policy="error",
                none_policy="pass-through",
            ),
        )

    x = ctx.wrap(x_layout)
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
            sumwxn = ak.operations.ak_sum._impl(
                x**n,
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
            sumwxn = ak.operations.ak_sum._impl(
                (x * weight) ** n,
                axis,
                keepdims,
                mask_identity,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
        return ctx.wrap(
            maybe_highlevel_to_lowlevel(sumwxn / sumw),
            highlevel=highlevel,
            allow_other=True,
        )
