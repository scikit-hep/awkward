# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("replace_substring_regex",)


@high_level_function(module="ak.str")
def replace_substring_regex(
    array,
    pattern,
    replacement,
    *,
    max_replacements=None,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str): Regular expression pattern to look for inside input values.
        replacement (str or bytes): What to replace the pattern with.
        max_replacements (None or int): If not None and not -1, limits the
            maximum number of replacements per string/bytestring, counting from
            the left.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Replaces non-overlapping subsequences of any string or bytestring-valued
    data that match a regular expression `pattern` with `replacement`.

    The `pattern` and `replacement` are scalars; they cannot be different
    for each string/bytestring in the sample.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.replace_substring_regex](https://arrow.apache.org/docs/python/generated/pyarrow.compute.replace_substring_regex.html)
    or
    [pyarrow.compute.replace_substring_regex](https://arrow.apache.org/docs/python/generated/pyarrow.compute.replace_substring_regex.html)
    on strings and bytestrings, respectively.

    See also: #ak.str.replace_substring_regex.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(
        array, pattern, replacement, max_replacements, highlevel, behavior, attrs
    )


def _impl(array, pattern, replacement, max_replacements, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("x")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = ak._do.recursively_apply(
        layout,
        ak.operations.str._get_ufunc_action(
            pc.replace_substring_regex,
            pc.replace_substring_regex,
            pattern,
            replacement,
            max_replacements=max_replacements,
        ),
    )

    return ctx.wrap(out, highlevel=highlevel)
