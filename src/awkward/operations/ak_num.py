# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._namedaxis import (
    _get_named_axis,
    _keep_named_axis,
    _named_axis_to_positional_axis,
)
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward._typing import Mapping
from awkward.errors import AxisError

__all__ = ("num",)

np = NumpyMetadata.instance()


@high_level_function()
def num(
    array,
    axis=1,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping | None = None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an array of integers specifying the number of elements at a
    particular level.

    For instance, given the following doubly nested `array`,

        >>> array = ak.Array([[[1.1, 2.2, 3.3],
        ...                    [],
        ...                    [4.4, 5.5],
        ...                    [6.6]
        ...                   ],
        ...                   [],
        ...                   [[7.7],
        ...                    [8.8, 9.9]]
        ...                   ])

    The number of elements in `axis=1` is

        >>> ak.num(array, axis=1)
        <Array [4, 0, 2] type='3 * int64'>

    and the number of elements at the next level down, `axis=2`, is

        >>> ak.num(array, axis=2)
        <Array [[3, 0, 2, 1], [], [1, 2]] type='3 * var * int64'>

    The `axis=0` case is special: it returns a scalar, the length of the array.

        >>> ak.num(array, axis=0)
        3

    This function is useful for ensuring that slices do not raise errors. For
    instance, suppose that we want to select the first element from each
    of the outermost nested lists of `array`. One of these lists is empty, so
    selecting the first element (`0`) would raise an error. However, if our
    first selection is `ak.num(array) > 0`, we are left with only those lists
    that *do* have a first element:

        >>> array[ak.num(array) > 0, 0]
        <Array [[1.1, 2.2, 3.3], [7.7]] type='2 * var * float64'>

    To keep a placeholder (None) in each place we do not want to select,
    consider using #ak.mask instead of a #ak.Array.__getitem__.

        >>> array.mask[ak.num(array) > 0][:, 0]
        <Array [[1.1, 2.2, 3.3], None, [7.7]] type='3 * option[var * float64]'>
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(
    array,
    axis,
    highlevel: bool,
    behavior: Mapping | None,
    attrs: Mapping | None,
):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    # Handle named axis
    named_axis = _get_named_axis(ctx)
    # Step 1: Normalize named axis to positional axis
    axis = _named_axis_to_positional_axis(named_axis, axis)
    # Step 2: propagate named axis from input to output,
    #   use strategy "keep one" (see: awkward._namedaxis)
    out_named_axis = _keep_named_axis(named_axis, axis)

    axis = regularize_axis(axis, none_allowed=False)

    if maybe_posaxis(layout, axis, 1) == 0:
        nplike = layout.backend.nplike
        if isinstance(layout, ak.record.Record):
            return nplike.asarray(nplike.shape_item_as_index(1))
        else:
            return nplike.asarray(nplike.shape_item_as_index(layout.length))

    def action(layout, depth, **kwargs):
        posaxis = maybe_posaxis(layout, axis, depth)

        if posaxis == depth and layout.is_list:
            return ak.contents.NumpyArray(layout.stops.data - layout.starts.data)

        elif layout.is_leaf:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")

    out = ak._do.recursively_apply(layout, action, numpy_to_regular=True)

    wrapped_out = ctx.wrap(
        out,
        highlevel=highlevel,
    )

    # propagate named axis to output
    return ak.operations.ak_with_named_axis._impl(
        wrapped_out,
        named_axis=out_named_axis,
        highlevel=highlevel,
        behavior=ctx.behavior,
        attrs=ctx.attrs,
    )
