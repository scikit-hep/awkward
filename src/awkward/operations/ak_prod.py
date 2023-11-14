# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("prod", "nanprod")

np = NumpyMetadata.instance()


@high_level_function()
def prod(
    array,
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
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Multiplies elements of `array` (many types supported, including all
    Awkward Arrays and Records). The identity of multiplication is `1` and it
    is usually not masked. This operation is the same as NumPy's
    [prod](https://docs.scipy.org/doc/numpy/reference/generated/numpy.prod.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    See also #ak.nanprod.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)


@high_level_function()
def nanprod(
    array,
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
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Like #ak.prod, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.prod(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.prod.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(
        ak.operations.ak_nan_to_none._impl(array, True, behavior, attrs),
        axis,
        keepdims,
        mask_identity,
        highlevel,
        behavior,
        attrs,
    )


def _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    reducer = ak._reducers.Prod()

    out = ak._do.reduce(
        layout,
        reducer,
        axis=axis,
        mask=mask_identity,
        keepdims=keepdims,
        behavior=ctx.behavior,
    )
    return ctx.wrap(out, highlevel=highlevel, allow_other=True)


@ak._connect.numpy.implements("prod")
def _nep_18_impl_prod(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    keepdims=False,
    initial=UNSUPPORTED,
    where=UNSUPPORTED,
):
    return prod(a, axis=axis, keepdims=keepdims)


@ak._connect.numpy.implements("nanprod")
def _nep_18_impl_nanprod(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    keepdims=False,
    initial=UNSUPPORTED,
    where=UNSUPPORTED,
):
    return nanprod(a, axis=axis, keepdims=keepdims)
