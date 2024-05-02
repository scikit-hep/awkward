# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("fill_none",)

np = NumpyMetadata.instance()


@high_level_function()
def fill_none(array, value, axis=-1, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        value: Data with which to replace None.
        axis (None or int): If None, replace all None values in the array
            with the given value; if an int, The dimension at which this
            operation is applied. The outermost dimension is `0`, followed
            by `1`, etc., and negative values count backward from the
            innermost: `-1` is the innermost  dimension, `-2` is the next
            level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Replaces missing values (None) with a given `value`.

    For example, in the following

        >>> array = ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

    The None values could be replaced with `0` by

        >>> ak.fill_none(array, 0)
        <Array [[1.1, 0, 2.2], [], [0, 3.3, 4.4]] type='3 * var * float64'>

    The replacement value doesn't strictly need the same type as the
    surrounding data. For example, the None values could also be replaced
    by a string.

        >>> ak.fill_none(array, "hi")
        <Array [[1.1, 'hi', 2.2], [], ['hi', ...]] type='3 * var * union[float64, s...'>

    The list content now has a union type:

        >>> ak.fill_none(array, "hi").type.show()
        3 * var * union[
            float64,
            string
        ]

    The values could be floating-point numbers or strings.
    """
    # Dispatch
    yield array, value

    # Implementation
    return _impl(array, value, axis, highlevel, behavior, attrs)


def _impl(array, value, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        array_layout, value_layout = ensure_same_backend(
            ctx.unwrap(array, allow_record=True, allow_unknown=False),
            ctx.unwrap(
                value,
                allow_record=True,
                allow_unknown=False,
                use_from_iter=True,
                primitive_policy="pass-through",
                string_policy="pass-through",
            ),
        )

    if isinstance(value_layout, ak.record.Record):
        value_layout = value_layout.array[value_layout.at : value_layout.at + 1]
    elif isinstance(value_layout, ak.contents.Content):
        value_layout = value_layout[np.newaxis, ...]
    else:
        # Now that we know `valuelayout` isn't a low-level type, we must have scalars
        # Thus, we can safely promote these scalars to a layout without
        # adding a new axis
        value_layout = ak.operations.to_layout(
            value,
            allow_record=True,
            allow_unknown=False,
            use_from_iter=True,
            primitive_policy="promote",
            string_policy="promote",
        ).to_backend(array_layout.backend)

    if axis is None:

        def action(layout, continuation, **kwargs):
            if layout.is_option:
                return ak._do.fill_none(continuation(), value_layout)

    else:

        def action(layout, depth, **kwargs):
            posaxis = maybe_posaxis(layout, axis, depth)
            if posaxis is not None and posaxis + 1 == depth:
                if layout.is_option:
                    return ak._do.fill_none(layout, value_layout)
                elif layout.is_union or layout.is_record or layout.is_indexed:
                    return None
                else:
                    return layout

            elif layout.is_leaf:
                raise AxisError(
                    f"axis={axis} exceeds the depth of this array ({depth})"
                )

    out = ak._do.recursively_apply(array_layout, action)
    return ctx.wrap(out, highlevel=highlevel)
