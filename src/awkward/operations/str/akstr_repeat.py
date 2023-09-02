# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("repeat",)

import numbers

import awkward as ak
from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata

cpu = NumpyBackend.instance()
typetracer = TypeTracerBackend.instance()
np = NumpyMetadata.instance()


@high_level_function(module="ak.str")
def repeat(array, num_repeats, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        num_repeats: (int, or an array of them to broadcast): number of times
            to repeat each element
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
    from awkward._connect.pyarrow import import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pc = import_pyarrow_compute("ak.str.repeat")

    behavior = behavior_of(array, num_repeats, behavior=behavior)
    backend = backend_of(array, num_repeats, coerce_to_common=True, default=cpu)
    layout = ak.operations.to_layout(array).to_backend(backend)

    num_repeats_layout = ak.operations.to_layout(num_repeats, allow_other=True)
    if isinstance(num_repeats_layout, ak.contents.Content):
        num_repeats_layout = num_repeats_layout.to_backend(backend)

        def action(inputs, **kwargs):
            if inputs[0].is_list and inputs[0].parameter("__array__") in (
                "string",
                "bytestring",
            ):
                if not (
                    inputs[1].is_numpy and np.issubdtype(inputs[1].dtype, np.integer)
                ):
                    raise TypeError(
                        "num_repeats must be an integer or broadcastable to integers"
                    )

                return (_apply_through_arrow(pc.binary_repeat, *inputs),)

        (out,) = ak._broadcasting.broadcast_and_apply(
            (layout, num_repeats_layout), action, behavior
        )

    else:
        if not isinstance(num_repeats, numbers.Integral):
            raise TypeError(
                "num_repeats must be an integer or broadcastable to integers"
            )

        def action(layout, **kwargs):
            if layout.is_list and layout.parameter("__array__") in (
                "string",
                "bytestring",
            ):
                return _apply_through_arrow(pc.binary_repeat, layout, num_repeats)

        out = ak._do.recursively_apply(layout, action, behavior)

    return wrap_layout(out, behavior, highlevel)
