# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
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

__all__ = ("nansum", "sum")

np = NumpyMetadata.instance()


@high_level_function()
def sum(
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

    Sums over `array` (many types supported, including all Awkward Arrays
    and Records). The identity of addition is `0` and it is usually not
    masked. This operation is the same as NumPy's
    [sum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, consider this `array`, in which all lists at a given dimension
    have the same length.

        >>> array = ak.Array([[ 0.1,  0.2,  0.3],
        ...                   [10.1, 10.2, 10.3],
        ...                   [20.1, 20.2, 20.3],
        ...                   [30.1, 30.2, 30.3]])

    A sum over `axis=-1` combines the inner lists, leaving one value per
    outer list:

        >>> ak.sum(array, axis=-1)
        <Array [0.6, 30.6, 60.6, 90.6] type='4 * float64'>

    while a sum over `axis=0` combines the outer lists, leaving one value
    per inner list:

        >>> ak.sum(array, axis=0)
        <Array [60.4, 60.8, 61.2] type='3 * float64'>

    Now with some values missing,

        >>> array = ak.Array([[ 0.1,  0.2      ],
        ...                   [10.1            ],
        ...                   [20.1, 20.2, 20.3],
        ...                   [30.1, 30.2      ]])

    The sum over `axis=-1` results in

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>

    and the sum over `axis=0` results in

        >>> ak.sum(array, axis=0)
        <Array [60.4, 50.6, 20.3] type='3 * float64'>

    How we ought to sum over the innermost lists is unambiguous, but for all
    other `axis` values, we must choose whether to align contents to the
    left before summing, to the right before summing, or something else.
    As suggested by the way the text has been aligned, we choose the
    left-alignment convention: the first `axis=0` result is the sum of all
    first elements

        60.4 = 0.1 + 10.1 + 20.1 + 30.1

    the second is the sum of all second elements

        50.6 = 0.2 + 20.2 + 30.2

    and the third is the sum of the only third element

        20.3 = 20.3

    The same is true if the values were None, rather than gaps:

        >>> array = ak.Array([[ 0.1,  0.2, None],
        ...                   [10.1, None, None],
        ...                   [20.1, 20.2, 20.3],
        ...                   [30.1, 30.2, None]])

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>
        >>> ak.sum(array, axis=0)
        <Array [60.4, 50.6, 20.3] type='3 * float64'>

    However, the missing value placeholder, None, allows us to align the
    remaining data differently:

        >>> array = ak.Array([[None,  0.1,  0.2],
        ...                   [None, None, 10.1],
        ...                   [20.1, 20.2, 20.3],
        ...                   [None, 30.1, 30.2]])

    Now the `axis=-1` result is the same but the `axis=0` result has changed:

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>
        >>> ak.sum(array, axis=0)
        <Array [20.1, 50.4, 60.8] type='3 * float64'>

    because

        20.1 = 20.1
        50.4 = 0.1 + 20.2 + 30.1
        60.8 = 0.2 + 10.1 + 20.3 + 30.2

    If, instead of missing numbers, we had missing lists,

        >>> array = ak.Array([[ 0.1,  0.2,  0.3],
        ...                   None,
        ...                   [20.1, 20.2, 20.3],
        ...                   [30.1, 30.2, 30.3]])

    then the placeholder would pass through the `axis=-1` sum because summing
    over the inner dimension shouldn't change the length of the outer
    dimension.

        >>> ak.sum(array, axis=-1)
        <Array [0.6, None, 60.6, 90.6] type='4 * ?float64'>

    However, the `axis=0` sum loses information about the None value.

        >>> ak.sum(array, axis=0)
        <Array [50.3, 50.6, 50.9] type='3 * float64'>

    which is

        50.3 = 0.1 + (None) + 20.1 + 30.1
        50.6 = 0.2 + (None) + 20.2 + 30.2
        50.9 = 0.3 + (None) + 20.3 + 30.3

    An `axis=0` sum would be reducing that information if it had not been
    None, anyway. If the None values were replaced with `0`, the result for
    `axis=0` would be the same. The result for `axis=-1` would not be the
    same because this None is in the `0` axis, not the axis that `axis=-1`
    sums over.

    The `keepdims` parameter ensures that the number of dimensions does not
    change: scalar results are put into new length-1 dimensions:

        >>> ak.sum(array, axis=-1, keepdims=True)
        <Array [[0.6], None, [60.6], [90.6]] type='4 * option[1 * float64]'>
        >>> ak.sum(array, axis=0, keepdims=True)
        <Array [[50.3, 50.6, 50.9]] type='1 * var * float64'>

    and `axis=None` ignores all None values and adds up everything in the
    array (`keepdims` has no effect).

        >>> ak.sum(array, axis=None)
        151.8

    The `mask_identity`, which has no equivalent in NumPy, inserts None in
    the output wherever a reduction takes place over zero elements. This is
    different from reductions that are otherwise equal to the identity or
    are equal to the identity by cancellation.

        >>> array = ak.Array([[2.2, 2.2], [4.4, -2.2, -2.2], [], [0.0]])
        >>> ak.sum(array, axis=-1)
        <Array [4.4, 0, 0, 0] type='4 * float64'>
        >>> ak.sum(array, axis=-1, mask_identity=True)
        <Array [4.4, 0, None, 0] type='4 * ?float64'>

    The third list is reduced to `0` if `mask_identity=False` because `0` is
    the identity of addition, but it is reduced to None if
    `mask_identity=True`.

    See also #ak.nansum.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)


@high_level_function()
def nansum(
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

    Like #ak.sum, but treating NaN ("not a number") values as missing.

    Equivalent to

        ak.sum(ak.nan_to_none(array))

    with all other arguments unchanged.

    See also #ak.sum.
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

    reducer = ak._reducers.Sum()

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


@ak._connect.numpy.implements("sum")
def _nep_18_impl_sum(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    keepdims=False,
    initial=UNSUPPORTED,
    where=UNSUPPORTED,
):
    return sum(a, axis=axis, keepdims=keepdims)


@ak._connect.numpy.implements("nansum")
def _nep_18_impl_nansum(
    a,
    axis=None,
    dtype=UNSUPPORTED,
    out=UNSUPPORTED,
    keepdims=False,
    initial=UNSUPPORTED,
    where=UNSUPPORTED,
):
    return nansum(a, axis=axis, keepdims=keepdims)
