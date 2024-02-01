# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("starts_with",)


@high_level_function(module="ak.str")
def starts_with(
    array, pattern, *, ignore_case=False, highlevel=True, behavior=None, attrs=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Substring pattern to test against the start
            of each string in `array`.
        ignore_case (bool): If True, perform a case-insensitive match;
            otherwise, the match is case-sensitive.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns True for every string in `array` if it starts with the given literal
    suffix `pattern`. Depending upon the value of `ignore_case`, the matching
    function will be case-insensitive.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.starts_with](https://arrow.apache.org/docs/python/generated/pyarrow.compute.starts_with.html).
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, pattern, ignore_case, highlevel, behavior, attrs)


def _impl(array, pattern, ignore_case, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.starts_with")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(
            array,
            allow_record=False,
            allow_unknown=False,
            primitive_policy="error",
            string_policy="as-characters",
        )
    apply = ak.operations.str._get_ufunc_action(
        pc.starts_with,
        pc.starts_with,
        bytestring_to_string=False,
        ignore_case=ignore_case,
        pattern=pattern,
    )
    out = ak._do.recursively_apply(layout, apply)
    return ctx.wrap(out, highlevel=highlevel)
