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

__all__ = ("count",)

np = NumpyMetadata.instance()


@high_level_function()
def count(
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

    Counts elements of `array` (many types supported, including all
    Awkward Arrays and Records). The identity of counting is `0` and it is
    usually not masked.

    This function has no analog in NumPy because counting values in a
    rectilinear array would only result in elements of the NumPy array's
    [shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html).

    However, for nested lists of variable dimension and missing values, the
    result of counting is non-trivial. For example, with this

        >>> array = ak.Array([[ 0.1,  0.2      ],
        ...                   [None, 10.2, None],
        ...                   None,
        ...                   [20.1, 20.2, 20.3],
        ...                   [30.1, 30.2      ]])

    the result of counting over the innermost dimension is

        >>> ak.count(array, axis=-1)
        <Array [2, 1, None, 3, 2] type='5 * ?int64'>

    the outermost dimension is

        >>> ak.count(array, axis=0)
        <Array [3, 4, 1] type='3 * int64'>

    and all dimensions is

        >>> ak.count(array, axis=None)
        8

    The gaps and None values are not counted, and if a None value occurs at
    a higher axis than the one being counted, it is kept as a placeholder
    so that the outer list length does not change.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    Note also that this function is different from #ak.num, which counts
    the number of values at a given depth, maintaining structure: #ak.num
    never counts across different lists the way that reducers do (#ak.num
    is not a reducer; #ak.count is). For the same `array`,

        >>> ak.num(array, axis=0)
        5
        >>> ak.num(array, axis=1)
        <Array [2, 3, None, 3, 2] type='5 * ?int64'>

    If it is desirable to include None values in #ak.count, use #ak.fill_none
    to turn the None values into something that would be counted.

    If it is desirable to exclude NaN ("not a number") values from #ak.count,
    use #ak.nan_to_none to turn them into None, which are not counted.
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

    reducer = ak._reducers.Count()

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
