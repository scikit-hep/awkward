# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._namedaxis import (
    _get_named_axis,
    _keep_named_axis,
    _named_axis_to_positional_axis,
    _remove_named_axis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("count_nonzero",)

np = NumpyMetadata.instance()


@high_level_function()
def count_nonzero(
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

    Counts nonzero elements of `array` (many types supported, including all
    Awkward Arrays and Records). The identity of counting is `0` and it is
    usually not masked. This operation is the same as NumPy's
    [count_nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    Following the same rules as other reducers, #ak.count_nonzero does not
    count None values. If it is desirable to count them, use #ak.fill_none
    to turn them into something that would be counted.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)


def _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    # Handle named axis
    named_axis = _get_named_axis(ctx)
    # Step 1: Normalize named axis to positional axis
    axis = _named_axis_to_positional_axis(named_axis, axis)
    # Step 2: propagate named axis from input to output,
    #   keepdims=True: use strategy "keep all" (see: awkward._namedaxis)
    #   keepdims=False: use strategy "remove one" (see: awkward._namedaxis)
    out_named_axis = _keep_named_axis(named_axis, None)
    if not keepdims:
        out_named_axis = _remove_named_axis(
            named_axis=out_named_axis,
            axis=axis,
            total=layout.minmax_depth[1],
        )

    axis = regularize_axis(axis, none_allowed=True)

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    reducer = ak._reducers.CountNonzero()

    out = ak._do.reduce(
        layout,
        reducer,
        axis=axis,
        mask=mask_identity,
        keepdims=keepdims,
        behavior=ctx.behavior,
    )

    wrapped_out = ctx.wrap(
        out,
        highlevel=highlevel,
        allow_other=True,
    )

    # propagate named axis to output
    return ak.operations.ak_with_named_axis._impl(
        wrapped_out,
        named_axis=out_named_axis,
        highlevel=highlevel,
        behavior=ctx.behavior,
        attrs=ctx.attrs,
    )


@ak._connect.numpy.implements("count_nonzero")
def _nep_18_impl(a, axis=None, *, keepdims=False):
    return count_nonzero(a, axis=axis, keepdims=keepdims)
