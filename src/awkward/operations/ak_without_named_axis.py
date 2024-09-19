# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._namedaxis import (
    _NamedAxisKey,
)
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("without_named_axis",)

np = NumpyMetadata.instance()


@high_level_function()
def without_named_axis(
    array,
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
    array in-place. If the new name is None, then an array without a name is
    returned.

    The records or tuples may be nested within multiple levels of nested lists.
    If records are nested within records, only the outermost are affected.

    Setting the `"__record__"` parameter makes it possible to add behaviors
    to the data; see #ak.Array and #ak.behavior for a more complete
    description.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False)

    return ctx.without_attr(key=_NamedAxisKey).wrap(
        layout,
        highlevel=highlevel,
        allow_other=True,
    )
