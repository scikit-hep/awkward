# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("materialize",)


@high_level_function()
def materialize(
    array,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array : Array-like data (anything #ak.to_layout recognizes).
            An array that may contain virtual buffers to be materialized.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Traverses the input array and materializes any virtual buffers.
    If the input array is not an #ak.Array or an #ak.contents.Content,
    it will just be cast to an #ak.Array.
    The buffers of the returned array are no longer `VirtualArray` objects even if there were any.
    They will become one of `numpy.ndarray`, `cupy.ndarray`, or `jax.numpy.ndarray` objects,
    depending on the array's backend.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(
            array, allow_record=False, primitive_policy="error", string_policy="error"
        )
    out = layout.materialize()
    return ctx.wrap(out, highlevel=highlevel)
