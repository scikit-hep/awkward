# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._namedaxis import _supports_named_axis, _check_valid_axis, _identity_named_axis, _remove_named_axis, _one_axis_to_positional_axis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import is_integer, regularize_axis

__all__ = ("argsort",)

np = NumpyMetadata.instance()


@high_level_function()
def argsort(
    array,
    axis=-1,
    *,
    ascending=True,
    stable=True,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        ascending (bool): If True, the first value in each sorted group
            will be smallest, the last value largest; if False, the order
            is from largest to smallest.
        stable (bool): If True, use a stable sorting algorithm; if False,
            use a sorting algorithm that is not guaranteed to be stable.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an array of integer indexes that would sort the array if applied
    as an integer-array slice.

    For example,

        >>> ak.argsort(ak.Array([[7.7, 5.5, 7.7], [], [2.2], [8.8, 2.2]]))
        <Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>

    The result of this function can be used to index other arrays with the
    same shape:

        >>> data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
        >>> index = ak.argsort(data)
        >>> index
        <Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>
        >>> data[index]
        <Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, ascending, stable, highlevel, behavior, attrs)


def _impl(array, axis, ascending, stable, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    out_named_axis = None
    if _supports_named_axis(ctx or {}) and _check_valid_axis(axis):
        # Handle named axis
        # Step 1: Normalize named axis to positional axis
        axis = _one_axis_to_positional_axis(axis, array.named_axis, array.positional_axis)

    # Step 2: propagate named axis from input to output,
    #   use strategy "keep all" (see: awkward._namedaxis)
    out_named_axis = _identity_named_axis(array.named_axis)


    if not isinstance(axis, int) and axis is not None:
        raise TypeError(f"'axis' must be an integer or None by now, not {axis!r}")

    out = ak._do.argsort(layout, axis, ascending, stable)

    wrapped_out = ctx.wrap(
        out,
        highlevel=highlevel,
    )

    if out_named_axis:
        # propagate named axis to output
        return ak.operations.ak_with_named_axis._impl(
            wrapped_out,
            named_axis=out_named_axis,
            highlevel=highlevel,
            behavior=ctx.behavior,
            attrs=ctx.attrs,
        )
    return wrapped_out


@ak._connect.numpy.implements("argsort")
def _nep_18_impl(a, axis=-1, kind=None, order=UNSUPPORTED):
    if kind is None:
        stable = False
    elif kind in ("stable", "mergesort"):
        stable = True
    elif kind in ("heapsort", "quicksort"):
        stable = False
    else:
        raise ValueError(
            f"unsupported value for 'kind' passed to overloaded NumPy function 'argsort': {kind!r}"
        )
    return argsort(a, axis=axis, stable=stable)
