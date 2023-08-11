# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("repeat",)

import numbers

import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


@high_level_function(module="ak.str")
def repeat(array, num_repeats, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        num_repeats: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Replaces any string-valued or bytestring-valued data with the same value
    repeated `num_repeats` times, which can be a scalar integer or a
    (broadcasted) array of integers.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.binary_repeat](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_repeat.html)
    or
    [pyarrow.compute.binary_repeat](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_repeat.html)
    on strings and bytestrings, respectively.
    """
    # Dispatch
    yield (array, num_repeats)

    # Implementation
    return _impl(array, num_repeats, highlevel, behavior)


def _impl(array, num_repeats, highlevel, behavior):
    import awkward._connect.pyarrow  # noqa: F401, I001
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow

    import pyarrow.compute as pc

    layout = ak.operations.to_layout(array)
    behavior = behavior_of(array, behavior=behavior)

    num_repeats_layout = ak.operations.to_layout(num_repeats, allow_other=True)

    if not isinstance(num_repeats_layout, ak.contents.Content):
        if not isinstance(num_repeats, numbers.Integral):
            raise TypeError(
                "num_repeats must be an integer or broadcastable to integers"
            )

        def action(layout, **kwargs):
            if layout.is_list and layout.parameter("__array__") in (
                "string",
                "bytestring",
            ):
                return from_arrow(
                    pc.binary_repeat(
                        to_arrow(layout, extensionarray=False), num_repeats
                    ),
                    highlevel=False,
                )

        out = ak._do.recursively_apply(layout, action, behavior)

    else:

        def action(inputs, **kwargs):
            if inputs[0].is_list and inputs[0].parameter("__array__") in (
                "string",
                "bytestring",
            ):
                if not inputs[1].is_numpy or not issubclass(
                    inputs[1].dtype.type, np.integer
                ):
                    raise TypeError(
                        "num_repeats must be an integer or broadcastable to integers"
                    )
                return (
                    from_arrow(
                        pc.binary_repeat(
                            to_arrow(inputs[0], extensionarray=False),
                            to_arrow(inputs[1], extensionarray=False),
                        ),
                        highlevel=False,
                    ),
                )

        out = ak._broadcasting.broadcast_and_apply(
            (layout, num_repeats_layout), action, behavior
        )
        assert isinstance(out, tuple) and len(out) == 1
        out = out[0]

    return wrap_layout(out, behavior, highlevel)
