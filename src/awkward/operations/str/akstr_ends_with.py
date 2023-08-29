# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("ends_with",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def ends_with(array, pattern, *, ignore_case=False, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Substring pattern to test against the ending
            of each string in `array`.
        ignore_case (bool): If True, perform a case-insensitive match;
            otherwise, the match is case-sensitive.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns True for every string in `array` if it ends with the given literal
    suffix `pattern`. Depending upon the value of `ignore_case`, the matching
    function will be case-insensitive.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.ends_with](https://arrow.apache.org/docs/python/generated/pyarrow.compute.ends_with.html).
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, pattern, ignore_case, highlevel, behavior)


def _impl(array, pattern, ignore_case, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.ends_with")
    layout = ak.to_layout(array, allow_record=False)
    behavior = behavior_of(array, behavior=behavior)
    apply = ak.operations.str._get_ufunc_action(
        pc.ends_with,
        pc.ends_with,
        bytestring_to_string=False,
        ignore_case=ignore_case,
        pattern=pattern,
    )
    out = ak._do.recursively_apply(layout, apply, behavior=behavior)
    return wrap_layout(out, highlevel=highlevel, behavior=behavior)
