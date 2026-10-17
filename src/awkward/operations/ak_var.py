# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak
from awkward._attrs import attrs_of_obj
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import (
    HighLevelContext,
    ensure_same_backend,
    maybe_highlevel_to_lowlevel,
    maybe_posaxis,
    promote_integral_to_float64,
)
from awkward._namedaxis import (
    NAMED_AXIS_KEY,
    _get_named_axis,
    _named_axis_to_positional_axis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("nanvar", "var")

np = NumpyMetadata.instance()


def _has_complex_leaf(layout) -> bool:
    """True if any NumpyArray leaf is complex (dtype kind 'c')."""
    found = False

    def action(node, **kwargs):
        nonlocal found
        if node.is_numpy and node.dtype.kind == "c":
            found = True
            return node
        return None

    ak._do.recursively_apply(layout, action, return_array=False)
    return found


@high_level_function()
def var(
    x,
    weight=None,
    ddof=0,
    axis=None,
    *,
    keepdims=False,
    mask_identity=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """Computes the variance over one or all levels of nesting.

    Many types are supported, including all Awkward Arrays and Records. The
    grouping is performed the same way as for reducers, though this operation is
    not a reducer and has no identity. It is the same as NumPy's
    [var](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the variance is calculated as::

        ak.sum((x - ak.mean(x))**2 * weight) / ak.sum(weight)

    If `ddof` is not zero, the above is further corrected by a factor of::

        ak.sum(weight) / (ak.sum(weight) - ddof)

    Even without `ddof`, #ak.var differs from #ak.moment with `n=2` because
    the mean is subtracted from all points before summing their squares.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.

    See also #ak.nanvar.

    Args:
        x: The data on which to compute the variance (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            variance."
        axis (None or int or str): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc; if a str, it is interpreted as the
            name of the axis which maps to an int if named axes are present.
            Named axes are attached to an array using #ak.with_named_axis and
            removed with #ak.without_named_axis; also see the
            [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
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

    Returns:
        The variance in each group of elements from `x`.
    """
    # Dispatch
    yield x, weight

    # Implementation
    return _impl(
        x, weight, ddof, axis, keepdims, mask_identity, highlevel, behavior, attrs
    )


@high_level_function()
def nanvar(
    x,
    weight=None,
    ddof=0,
    axis=None,
    *,
    keepdims=False,
    mask_identity=True,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """Computes the variance, treating NaN values as missing.

    Equivalent to::

        ak.var(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.var.

    Args:
        x: The data on which to compute the variance (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            variance."
        axis (None or int or str): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc; if a str, it is interpreted as the
            name of the axis which maps to an int if named axes are present.
            Named axes are attached to an array using #ak.with_named_axis and
            removed with #ak.without_named_axis; also see the
            [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
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

    Returns:
        Like #ak.var, but treating NaN ("not a number") values as missing.
    """
    # Dispatch
    yield x, weight

    # Implementation
    if weight is not None:
        weight = ak.operations.ak_nan_to_none._impl(weight, True, behavior, attrs)

    return _impl(
        ak.operations.ak_nan_to_none._impl(x, True, behavior, attrs),
        weight,
        ddof,
        axis,
        keepdims,
        mask_identity,
        highlevel,
        behavior,
        attrs,
    )


def _impl(x, weight, ddof, axis, keepdims, mask_identity, highlevel, behavior, attrs):
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

    # Handle named axis
    named_axis = _get_named_axis(ctx)
    # Step 1: Normalize named axis to positional axis
    axis = _named_axis_to_positional_axis(named_axis, axis)
    axis = regularize_axis(axis, none_allowed=True)

    kw = {
        "keepdims": True,
        "mask_identity": True,
        "highlevel": True,
        "behavior": ctx.behavior,
        "attrs": ctx.attrs,
    }
    is_complex = _has_complex_leaf(x.layout)

    with np.errstate(invalid="ignore", divide="ignore"):
        if weight is None:
            sumw = ak.operations.ak_count._impl(x, axis, **kw)
        else:
            sumw = ak.operations.ak_sum._impl(x * 0 + weight, axis, **kw)

        # Fuse only for the *innermost* axis. There the reduce descends through
        # the outer lists without transposing, so each innermost sublist becomes a
        # bin in depth-first order -- exactly the order of
        # ravel(mean(axis, keepdims)), so `means_flat` aligns bin-for-bin. For a
        # non-innermost axis the descent rearranges the content (a carry/transpose)
        # and that alignment no longer holds, so those axes use the (correct)
        # two-pass/broadcast path below instead.
        depth_min, depth_max = x.layout.minmax_depth
        posaxis = maybe_posaxis(x.layout, axis, 1) if axis is not None else None
        if (
            weight is None
            and not is_complex
            and axis is not None
            and depth_min == depth_max
            and posaxis == depth_max - 1
            and ak.backend(x) in ("cpu", "cuda")
        ):
            # Fused centered sum-of-squares: Sigma (x - mean)**2 per segment in a
            # single pass -- no materialised deviation buffer and no back-broadcast
            # of the mean (the dominant cost of the plain two-pass at the innermost
            # axis). Overflow-safe and stable (deviations are formed in float64
            # inside the kernel).
            means_flat = ak.operations.ak_ravel._impl(
                ak.operations.ak_mean._impl(
                    x,
                    None,
                    axis,
                    keepdims=True,
                    mask_identity=False,
                    highlevel=True,
                    behavior=ctx.behavior,
                    attrs=ctx.attrs,
                ),
                highlevel=True,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
            sumwxx = ak.operations.ak_centered_sumofsquares._impl(
                x, means_flat, axis, **kw
            )
            out = sumwxx / sumw
        else:
            # Two-pass, like NumPy and ak.covar: centre on the (float64) mean, then
            # sum the squared deviations. Numerically stable (the one-pass
            # E[x**2]-E[x]**2 form catastrophically cancels) and overflow-safe.
            # Used for weighted, complex, axis=None, a non-innermost axis, and
            # non-cpu/cuda backends (typetracer, jax).
            # Centring needs `x - xmean` to broadcast, which is undefined when the
            # reduced axis is not the innermost one of a *ragged* array; there the
            # one-pass form is used instead.
            xmean = ak.operations.ak_mean._impl(x, weight, axis, **kw)
            # Strip named axes so the subtraction is not rejected by the named-axis
            # check (the output's named axis is propagated from `x`, not `xmean`).
            xmean = ak.operations.ak_without_named_axis._impl(
                xmean, highlevel=True, behavior=ctx.behavior, attrs=ctx.attrs
            )
            if axis is None:
                # axis=None collapses to a scalar mean; subtract the scalar so
                # centring hits the flat content instead of a slow ragged
                # broadcast. (Empty -> None mean -> one-pass.)
                m_scalar = xmean[(0,) * xmean.ndim]
                dev = None if m_scalar is None else (x - m_scalar)
            else:
                try:
                    dev = x - xmean
                except ValueError:
                    dev = None

            if dev is not None:
                if is_complex:
                    # Variance of complex data is E[|x - mean|**2], a real number.
                    squared_dev = abs(dev) ** 2
                    if weight is not None:
                        squared_dev = squared_dev * weight
                    sumwxx = ak.operations.ak_sum._impl(squared_dev, axis, **kw)
                elif weight is None:
                    sumwxx = ak.operations.ak_sumofsquares._impl(dev, axis, **kw)
                else:
                    sumwxx = ak.operations.ak_sum._impl(weight * dev * dev, axis, **kw)
                out = sumwxx / sumw
            else:
                # One-pass fallback (non-innermost ragged axis). Complex variance
                # is E[|x|**2] - |E[x]|**2 (real), matching the innermost path.
                if weight is None:
                    sumwx = ak.operations.ak_sum._impl(x, axis, dtype=np.float64, **kw)
                    if is_complex:
                        sumwxx = ak.operations.ak_sum._impl(abs(x) ** 2, axis, **kw)
                    else:
                        sumwxx = ak.operations.ak_sumofsquares._impl(x, axis, **kw)
                else:
                    xp = x if is_complex else promote_integral_to_float64(x)
                    sumwx = ak.operations.ak_sum._impl(
                        xp * weight, axis, dtype=np.float64, **kw
                    )
                    if is_complex:
                        sumwxx = ak.operations.ak_sum._impl(
                            abs(xp) ** 2 * weight, axis, **kw
                        )
                    else:
                        sumwxx = ak.operations.ak_sum._impl(
                            xp * xp * weight, axis, **kw
                        )
                mean = sumwx / sumw
                out = sumwxx / sumw - (abs(mean) ** 2 if is_complex else mean * mean)

        if ddof != 0:
            out = out * (sumw / (sumw - ddof))

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

        wrapped = ctx.without_attr(NAMED_AXIS_KEY).wrap(
            maybe_highlevel_to_lowlevel(out),
            highlevel=highlevel,
            allow_other=True,
        )

        # propagate named axis to output
        return ak.operations.ak_with_named_axis._impl(
            wrapped,
            named_axis=_get_named_axis(attrs_of_obj(out), allow_any=True),
            highlevel=highlevel,
            behavior=None,
            attrs=None,
        )


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
