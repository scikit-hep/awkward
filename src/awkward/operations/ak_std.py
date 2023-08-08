# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("std",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import maybe_posaxis
from awkward._nplikes import ufuncs
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import regularize_axis

np = NumpyMetadata.instance()


@high_level_function()
def std(x, weight=None, ddof=0, axis=None, *, keepdims=False, mask_identity=False):
    """
    Args:
        x: The data on which to compute the standard deviation (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            standard deviation."
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

    Computes the standard deviation in each group of elements from `x`
    (many types supported, including all Awkward Arrays and Records). The
    grouping is performed the same way as for reducers, though this operation
    is not a reducer and has no identity. It is the same as NumPy's
    [std](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the standard deviation is
    calculated as

        np.sqrt(ak.var(x, weight))

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.

    See also #ak.nanstd.
    """
    # Dispatch
    yield x, weight

    # Implementation
    return _impl(x, weight, ddof, axis, keepdims, mask_identity)


@high_level_function()
def nanstd(x, weight=None, ddof=0, axis=None, *, keepdims=False, mask_identity=True):
    """
    Args:
        x: The data on which to compute the standard deviation (anything #ak.to_layout recognizes).
        weight: Data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            standard deviation."
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

    Like #ak.std, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.std(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.std.
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
        out = ufuncs.sqrt(
            ak.operations.ak_var._impl(
                x,
                weight,
                ddof,
                axis,
                keepdims=True,
                mask_identity=True,
            )
        )

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


@ak._connect.numpy.implements("std")
def _nep_18_impl_std(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    ddof=0,
    keepdims=False,
    *,
    where=UNSUPPORTED,
):
    return std(a, axis=axis, keepdims=keepdims, ddof=ddof)


@ak._connect.numpy.implements("nanstd")
def _nep_18_impl_nanstd(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    ddof=0,
    keepdims=False,
    *,
    where=UNSUPPORTED,
):
    return nanstd(a, axis=axis, keepdims=keepdims, ddof=ddof)
