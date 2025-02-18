# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
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
        array : array_like
            An array with possible virtual buffers materialize.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Traverses the input array and materializes any virtual buffers.
    The buffers of the returned array are no longer `VirtualArray` objects.
    They will become either `numpy.ndarray` or `cupy.ndarray` objects depending on the array's backend.
    Possible inputs that will be traversed are instances of #ak.Array, #ak.Record, #ak.contents.Content, and #ak.record.Record.
    All other types of inputs will be returned as is.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    if not isinstance(
        array,
        (
            ak.highlevel.Array,
            ak.highlevel.Record,
            ak.contents.Content,
            ak.record.Record,
        ),
    ):
        return array

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=True, allow_unknown=False)
    out = layout.materialize()
    return ctx.wrap(out, highlevel=highlevel)
