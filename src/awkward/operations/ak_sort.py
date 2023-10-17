# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("sort",)

np = NumpyMetadata.instance()


@high_level_function()
def sort(
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

    Returns a sorted array.

    For example,

        >>> ak.sort(ak.Array([[7, 5, 7], [], [2], [8, 2]]))
        <Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, ascending, stable, highlevel, behavior, attrs)


def _impl(array, axis, ascending, stable, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    out = ak._do.sort(layout, axis, ascending, stable)
    return ctx.wrap(out, highlevel=highlevel)


@ak._connect.numpy.implements("sort")
def _nep_18_impl(a, axis=-1, kind=None, order=UNSUPPORTED):
    if kind is None:
        stable = False
    elif kind in ("stable", "mergesort"):
        stable = True
    elif kind in ("heapsort", "quicksort"):
        stable = False
    else:
        raise ValueError(
            f"unsupported value for 'kind' passed to overloaded NumPy function 'sort': {kind!r}"
        )
    return sort(a, axis=axis, stable=stable)
