# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._namedaxis import (
    NAMED_AXIS_KEY,
    AxisMapping,
    AxisTuple,
    _prepare_named_axis_for_attrs,
)
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("with_named_axis",)

np = NumpyMetadata.instance()


@high_level_function()
def with_named_axis(
    array,
    named_axis: AxisTuple | AxisMapping,
    *,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        named_axis: AxisTuple | AxisMapping: Names to give to the array axis; this assigns
            the `"__named_axis__"` attr. If None, any existing name is unset.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new name. This function does not change the
    array in-place. If the new name is None, then the array is returned as it is.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, named_axis, highlevel, behavior, attrs)


def _impl(array, named_axis, highlevel, behavior, attrs):
    # Named axis handling
    if not named_axis:  # no-op, e.g. named_axis is None, (), {}
        return array

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=True)

    _named_axis = _prepare_named_axis_for_attrs(
        named_axis=named_axis,
        ndim=layout.minmax_depth[1],
    )
    # now we're good, set the named axis
    return ctx.with_attr(
        key=NAMED_AXIS_KEY,
        value=_named_axis,
    ).wrap(
        layout,
        highlevel=highlevel,
        allow_other=True,
    )
