# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("merge_option_of_records",)

np = NumpyMetadata.instance()


@high_level_function()
def merge_option_of_records(
    array, axis=-1, *, highlevel=True, behavior=None, attrs=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the  innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Simplifies options of records, e.g.

        >>> array = ak.Array([None, {"a": 1}, {"a": 2}])

    into records of options, i.e.

        >>> ak.merge_option_of_records(array)
        <Array [{a: None}, {a: 1}, {a: 2}] type='3 * {a: ?int64}'>
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    # First, normalise type-invsible "index-of-records" to "record-of-index"
    def apply_displace_index(layout, backend, **kwargs):
        if (layout.is_indexed and not layout.is_option) and layout.content.is_record:
            record = layout.content

            # Transpose index-of-record to record-of-index
            return ak.contents.RecordArray(
                [
                    ak.contents.IndexedArray.simplified(
                        layout.index, c, parameters=layout._parameters
                    )
                    for c in record.contents
                ],
                record.fields,
                layout.length,
                backend=backend,
            )

    layout = ak._do.recursively_apply(layout, apply_displace_index)

    def apply(layout, depth, backend, **kwargs):
        posaxis = maybe_posaxis(layout, axis, depth)
        if depth < posaxis + 1 and layout.is_leaf:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
        elif depth == posaxis + 1 and layout.is_option and layout.content.is_record:
            layout = layout.to_IndexedOptionArray64()

            record = layout.content
            # Transpose option-of-record to record-of-option
            return ak.contents.RecordArray(
                [
                    ak.contents.IndexedOptionArray.simplified(
                        layout.index, c, parameters=layout._parameters
                    )
                    for c in record.contents
                ],
                record.fields,
                layout.length,
                backend=backend,
            )

    out = ak._do.recursively_apply(layout, apply)
    return ctx.wrap(out, highlevel=highlevel)
