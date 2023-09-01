# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("slice",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def slice(array, start, stop=None, step=1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        start (int): Index to start slicing at (inclusive).
        stop (None or int): Index to stop slicing at (exclusive). If not given,
            slicing will stop at the end.
        step (int): Slice step.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Replaces any string or bytestring-valued data with a slice between `start`
    and `stop` indexes; `start` is inclusive and `stop` is exclusive and both
    are 0-indexed.

    For strings, `start` and `stop` are measured in Unicode characters; for
    bytestrings, `start` and `stop` are measured in bytes.

    The `start`, `stop`, and `replacement` are scalars; they cannot be
    different for each string/bytestring in the sample.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_slice_codeunits](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_slice_codeunits.html)
    or performs a literal slice on strings and bytestrings, respectively.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, start, stop, step, highlevel, behavior)


def _impl(array, start, stop, step, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pc = import_pyarrow_compute("ak.str.slice")

    behavior = behavior_of(array, behavior=behavior)

    def action(layout, **absorb):
        if layout.is_list and layout.parameter("__array__") == "string":
            return _apply_through_arrow(
                pc.utf8_slice_codeunits, layout, start, stop, step
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            return layout[:, start:stop:step]

    out = ak._do.recursively_apply(ak.operations.to_layout(array), action, behavior)

    return wrap_layout(out, behavior, highlevel)
