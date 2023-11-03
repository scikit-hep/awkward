# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("fill_none",)

import awkward as ak
from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import maybe_posaxis, wrap_layout
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

np = NumpyMetadata.instance()
cpu = NumpyBackend.instance()


@high_level_function()
def fill_none(array, value, axis=-1, *, highlevel=True, behavior=None):
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
    return _impl(array, value, axis, highlevel, behavior)


def _impl(array, value, axis, highlevel, behavior):
    axis = regularize_axis(axis)
    behavior = behavior_of(array, value, behavior=behavior)
    # Ensure a common backend exists
    backend = backend_of(array, value, default=cpu)
    arraylayout = ak.operations.to_layout(
        array, allow_record=True, allow_unknown=False
    ).to_backend(backend)
    valuelayout = ak.operations.to_layout(
        value,
        allow_record=True,
        allow_unknown=False,
        use_from_iter=True,
        primitive_policy="pass-through",
        string_policy="pass-through",
    )

    if isinstance(valuelayout, ak.record.Record):
        valuelayout = valuelayout.array[valuelayout.at : valuelayout.at + 1]
    elif isinstance(valuelayout, ak.contents.Content):
        valuelayout = valuelayout[np.newaxis, ...]
    else:
        # Now that we know `valuelayout` isn't a low-level type, we must have scalars
        # Thus, we can safely promote these scalars to a layout without
        # adding a new axis
        valuelayout = ak.operations.to_layout(
            value,
            allow_record=True,
            allow_unknown=False,
            use_from_iter=True,
            primitive_policy="promote",
            string_policy="promote",
        )
    valuelayout = valuelayout.to_backend(backend)

    if axis is None:

        def action(layout, continuation, **kwargs):
            if layout.is_option:
                return ak._do.fill_none(continuation(), valuelayout)

    else:

        def action(layout, depth, **kwargs):
            posaxis = maybe_posaxis(layout, axis, depth)
            if posaxis is not None and posaxis + 1 == depth:
                if layout.is_option:
                    return ak._do.fill_none(layout, valuelayout)
                elif layout.is_union or layout.is_record or layout.is_indexed:
                    return None
                else:
                    return layout

            elif layout.is_leaf:
                raise AxisError(
                    f"axis={axis} exceeds the depth of this array ({depth})"
                )

    out = ak._do.recursively_apply(arraylayout, action, behavior)
    return wrap_layout(out, behavior, highlevel)
