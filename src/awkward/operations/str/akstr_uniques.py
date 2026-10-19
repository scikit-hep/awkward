# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Unique string values for each final string-list axis."""

from __future__ import annotations

from awkward._dispatch import high_level_function

__all__ = ("uniques",)


@high_level_function(module="ak.str")
def uniques(array, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns one copy of each distinct value in an array of strings or
    bytestrings. For nested lists of strings, this is done separately in each
    innermost list and the outer nesting is preserved. A flat array of strings
    is treated as a single list.

    Values are returned in the order in which they first appear. Missing
    values (None) are skipped, so they never appear in the output. Categorical
    strings are returned as plain strings.

    Strings inside a record or union cannot be deduplicated without changing
    the length of that record or union, so they raise a TypeError.

    If `array` contains no string or bytestring data, this function returns it
    unchanged.

    Requires the pyarrow library and follows
    [pyarrow.compute.unique](https://arrow.apache.org/docs/python/generated/pyarrow.compute.unique.html).
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    from awkward.operations.str import _distinct_strings

    return _distinct_strings(array, False, "ak.str.uniques", highlevel, behavior, attrs)
