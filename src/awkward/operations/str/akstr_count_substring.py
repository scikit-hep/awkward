# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("count_substring",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def count_substring(
    array, pattern, *, ignore_case=False, highlevel=True, behavior=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Substring pattern to count for each string in
            `array`.
        ignore_case (bool): If True, perform a case-insensitive match;
            otherwise, the match is case-sensitive.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Counts the number of occurrences of the given literal `pattern` in every
    string in `array`. Depending upon the value of `ignore_case`, the matching
    function will be case-insensitive.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.count_substring](https://arrow.apache.org/docs/python/generated/pyarrow.compute.count_substring.html).

    See also: #ak.str.count_substring_regex.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, pattern, ignore_case, highlevel, behavior)


def _impl(array, pattern, ignore_case, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.count_substring")
    layout = ak.to_layout(array, allow_record=False)
    behavior = behavior_of(array, behavior=behavior)
    apply = ak.operations.str._get_ufunc_action(
        pc.count_substring,
        pc.count_substring,
        bytestring_to_string=False,
        ignore_case=ignore_case,
        pattern=pattern,
    )
    out = ak._do.recursively_apply(layout, apply, behavior=behavior)

    return wrap_layout(out, highlevel=highlevel, behavior=behavior)
