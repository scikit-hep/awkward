# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("replace_slice",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def replace_slice(array, start, stop, replacement, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        start (int): Index to start slicing at (inclusive).
        stop (int): Index to stop slicing at (exclusive).
        replacement (str or bytes): What to replace the slice with.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Replaces slices of any string or bytestring-valued data with `replacement`
    between `start` and `stop` indexes; `start` is inclusive and `stop` is
    exclusive and both are 0-indexed.

    For strings, `start` and `stop` are measured in Unicode characters; for
    bytestrings, `start` and `stop` are measured in bytes.

    The `start`, `stop`, and `replacement` are scalars; they cannot be
    different for each string/bytestring in the sample.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_replace_slice](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_replace_slice.html)
    or
    [pyarrow.compute.binary_replace_slice](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_replace_slice.html)
    on strings and bytestrings, respectively.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, start, stop, replacement, highlevel, behavior)


def _impl(array, start, stop, replacement, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.replace_slice")
    behavior = behavior_of(array, behavior=behavior)

    out = ak._do.recursively_apply(
        ak.operations.to_layout(array),
        ak.operations.str._get_ufunc_action(
            pc.utf8_replace_slice, pc.binary_replace_slice, start, stop, replacement
        ),
        behavior,
    )

    return wrap_layout(out, behavior, highlevel)
