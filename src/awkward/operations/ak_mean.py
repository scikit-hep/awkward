# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import (
    HighLevelContext,
    ensure_same_backend,
    maybe_highlevel_to_lowlevel,
    maybe_posaxis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("mean", "nanmean")

np = NumpyMetadata.instance()


@high_level_function()
def mean(
    x,
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
        x: The data on which to compute the mean (anything #ak.to_layout recognizes).
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

    Computes the mean in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity. It is the same as NumPy's
    [mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the mean is calculated as

        ak.sum(x*weight) / ak.sum(weight)

    For example, with an `array` like

        >>> array = ak.Array([[0, 1, 2, 3],
                              [          ],
                              [4, 5      ]])

    The mean of the innermost lists is

        >>> ak.mean(array, axis=-1)
        <Array [1.5, nan, 4.5] type='3 * float64'>

    because there are three lists, the first has mean `1.5`, the second is
    empty, and the third has mean `4.5`.

    The mean of the outermost lists is

        >>> ak.mean(array, axis=0)
        <Array [2, 3, 2, 3] type='4 * float64'>

    because the longest list has length 4, the mean of `0` and `4` is `2.0`,
    the mean of `1` and `5` is `3.0`, the mean of `2` (by itself) is `2.0`,
    and the mean of `3` (by itself) is `3.0`. This follows the same grouping
    behavior as reducers.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers.

    See also #ak.nanmean.
    """
    # Dispatch
    yield x, weight

    # Implementation
    return _impl(x, weight, axis, keepdims, mask_identity, highlevel, behavior, attrs)


@high_level_function()
def nanmean(
    x,
    weight=None,
    axis=None,
    *,
    keepdims=False,
    mask_identity=True,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        x: The data on which to compute the mean (anything #ak.to_layout recognizes).
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

    Like #ak.mean, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.mean(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.mean.
    """
    # Dispatch
    yield x, weight

    if weight is not None:
        weight = ak.operations.ak_nan_to_none._impl(weight, True, behavior, attrs)

    return _impl(
        ak.operations.ak_nan_to_none._impl(x, False, behavior, attrs),
        weight,
        axis,
        keepdims,
        mask_identity,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )


def _impl(x, weight, axis, keepdims, mask_identity, highlevel, behavior, attrs):
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
                keepdims=True,
                mask_identity=True,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwx = ak.operations.ak_sum._impl(
                x,
                axis,
                keepdims=True,
                mask_identity=True,
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
                keepdims=True,
                mask_identity=True,
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )

        out = sumwx / sumw

        if not mask_identity:
            out = ak.operations.fill_none(
                out,
                np.nan,
                axis=-1,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
                highlevel=True,
            )

        if axis is None:
            if not keepdims:
                out = out[(0,) * out.ndim]
        else:
            if not keepdims:
                posaxis = maybe_posaxis(out.layout, axis, 1)
                out = out[(slice(None, None),) * posaxis + (0,)]

        return ctx.wrap(
            maybe_highlevel_to_lowlevel(out), highlevel=highlevel, allow_other=True
        )


@ak._connect.numpy.implements("mean")
def _nep_18_impl_mean(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    keepdims=False,
    *,
    where=UNSUPPORTED,
):
    return mean(a, axis=axis, keepdims=keepdims)


@ak._connect.numpy.implements("nanmean")
def _nep_18_impl_nanmean(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    keepdims=False,
    *,
    where=UNSUPPORTED,
):
    return nanmean(a, axis=axis, keepdims=keepdims)
