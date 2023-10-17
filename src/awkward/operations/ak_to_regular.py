# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("to_regular",)

np = NumpyMetadata.instance()


@high_level_function()
def to_regular(array, axis=1, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int or None): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc. If None, convert all
            variable dimensions into regular ones or raise a ValueError if that
            is not possible.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts a variable-length axis into a regular one, if possible.

        >>> irregular = ak.from_iter(np.arange(2*3*5).reshape(2, 3, 5))
        >>> irregular.type.show()
        2 * var * var * int64
        >>> ak.to_regular(irregular).type.show()
        2 * 3 * var * int64
        >>> ak.to_regular(irregular, axis=2).type.show()
        2 * var * 5 * int64
        >>> ak.to_regular(irregular, axis=-1).type.show()
        2 * var * 5 * int64

    But truly irregular data cannot be converted.

        >>> ak.to_regular(ak.Array([[1, 2, 3], [], [4, 5]]))
        ValueError: while calling
            ak.to_regular(
                array = <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>
                axis = 1
                highlevel = True
                behavior = None
            )
        Error details: cannot convert to RegularArray because subarray lengths are not regular

    See also #ak.from_regular.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    if axis is None:

        def action(layout, continuation, **kwargs):
            if layout.is_list:
                return continuation().to_RegularArray()

        out = ak._do.recursively_apply(layout, action)

    elif maybe_posaxis(layout, axis, 1) == 0:
        out = layout  # the top-level can only be regular (ArrayType)

    else:

        def action(layout, depth, **kwargs):
            posaxis = maybe_posaxis(layout, axis, depth)
            if posaxis == depth and layout.is_list:
                return layout.to_RegularArray()

            elif layout.is_leaf:
                raise AxisError(
                    f"axis={axis} exceeds the depth of this array ({depth})"
                )

        out = ak._do.recursively_apply(layout, action)

    return ctx.wrap(out, highlevel=highlevel)
