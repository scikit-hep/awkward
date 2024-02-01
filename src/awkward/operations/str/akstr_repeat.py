# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numbers

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("repeat",)

typetracer = TypeTracerBackend.instance()
np = NumpyMetadata.instance()


@high_level_function(module="ak.str")
def repeat(array, num_repeats, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        num_repeats: (int, or an array of them to broadcast): number of times
            to repeat each element
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
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
    yield array, num_repeats

    # Implementation
    return _impl(array, num_repeats, highlevel, behavior, attrs)


def _impl(array, num_repeats, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pc = import_pyarrow_compute("ak.str.repeat")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout, num_repeats_layout = ensure_same_backend(
            ctx.unwrap(array, allow_record=False),
            ctx.unwrap(
                num_repeats, allow_record=False, primitive_policy="pass-through"
            ),
        )
    if isinstance(num_repeats_layout, ak.contents.Content):

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
            (layout, num_repeats_layout), action
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

        out = ak._do.recursively_apply(layout, action)

    return ctx.wrap(out, highlevel=highlevel)
