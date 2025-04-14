# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._namedaxis import (
    NAMED_AXIS_KEY,
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
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) without named axes. This function does not change the
    array in-place.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(
            array,
            allow_record=True,
            primitive_policy="pass-through",
            none_policy="pass-through",
            string_policy="pass-through",
        )

    return ctx.without_attr(key=NAMED_AXIS_KEY).wrap(
        layout,
        highlevel=highlevel,
        allow_other=True,
    )
