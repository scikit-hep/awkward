# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("drop_none",)

np = NumpyMetadata.instance()


@high_level_function()
def drop_none(array, axis=None, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Data in which to remove Nones.
        axis (None or int): If None, the operation drops Nones at all levels of
            nesting, returning an array of the same dimension, but without Nones.
            Otherwise, it drops Nones at a specified depth.
            The outermost dimension is `0`, followed by `1`, etc.,
            and negative values count backward from the innermost: `-1` is the
            innermost dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Removes missing values (None) from a given array.

    For example, in the following `array`,

        >>> array = ak.Array([[[0]], [[None]], [[1], None], [[2, None]]])

    The None value will be removed, resulting in

        >>> ak.drop_none(array)
        <Array [[[0]], [[]], [[1]], [[2]]] type='4 * var * var * int64'>

    The default axis is None, however an axis can be specified:

        >>> ak.drop_none(array, axis=1)
        <Array [[[0]], [[None]], [[1]], [[2, None]]] type='4 * var * var * ?int64'>

    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _drop_none_if_list(layout):
    if layout.is_list:
        # only drop nones at list level in the recursion; this way ListArray -> ListOffsetArray with unprojected optiontype -> avoid offset mismatch
        return layout.drop_none()
    else:
        return layout


def _impl(array, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    if axis is None:
        # if the outer layout is_option, drop_nones without affecting offsets
        if layout.is_option:
            layout = layout.drop_none()

        def action(layout, continuation, **kwargs):
            return _drop_none_if_list(continuation())

    else:
        max_axis = layout.branch_depth[1] - 1
        if axis > max_axis:
            raise AxisError(f"axis={axis} exceeds the depth ({max_axis}) of this array")

        def recompute_offsets(layout, depth, **kwargs):
            posaxis = maybe_posaxis(layout, axis, depth)
            if (
                posaxis == 0
                and posaxis == depth - 1
                or posaxis == depth
                and layout.is_list
            ):
                none_indexes = options["none_indexes"].pop(0)
                out = layout._rebuild_without_nones(none_indexes, layout.content)
                return out

        def action(layout, depth, **kwargs):
            if layout.is_record:
                posaxises = {maybe_posaxis(x, axis, depth) for x in layout.contents}
                if len(posaxises) > 1 and any(x < depth for x in posaxises):
                    raise AxisError(
                        f"axis={axis} implies different levels in records that might require part of a record to be dropped, which is impossible"
                    )
            posaxis = maybe_posaxis(layout, axis, depth)
            if posaxis == 0:
                if not layout.is_option:
                    return layout
                else:
                    return layout.drop_none()
            if posaxis == depth - 1 and layout.is_option:
                _, _, none_indexes = layout._nextcarry_outindex()
                options["none_indexes"].append(none_indexes)
                return layout.drop_none()
            if posaxis == depth - 1 and layout.is_list and layout.content.is_option:
                return layout.drop_none()

    options = {"none_indexes": []}
    out = ak._do.recursively_apply(layout, action, depth_context=options)

    if len(options["none_indexes"]) > 0:
        out = ak._do.recursively_apply(out, recompute_offsets, depth_context=options)

    return ctx.wrap(out, highlevel=highlevel)
