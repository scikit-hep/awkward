# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("split_pattern_regex",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function
def split_pattern_regex(
    array, pattern, max_splits=None, reverse=False, *, highlevel=True, behavior=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Regular expression of characters/bytes to split on.
        max_splits (None or int): Maximum number of splits for each input value. If None, unlimited.
        reverse (bool): If True, start splitting from the end of each input value; otherwise, start splitting
            from the beginning of each value. This flag only has an effect if `max_splits` is not None.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Splits any string or bytestring-valued data into a list of substrings according to the given regular expression.

    Note: this function does not raise an error if the `array` does not contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.split_pattern](https://arrow.apache.org/docs/python/generated/pyarrow.compute.split_pattern.html).
    """
    # Dispatch
    yield (array, pattern, max_splits, reverse)

    # Implementation
    return _impl(array, pattern, max_splits, reverse, highlevel, behavior)


def _impl(array, pattern, max_splits, reverse, highlevel, behavior):
    import awkward._connect.pyarrow  # noqa: F401, I001

    import pyarrow.compute as pc

    behavior = behavior_of(array, behavior=behavior)
    action = ak.operations.str._get_split_action(
        pc.split_pattern_regex,
        pc.split_pattern_regex,
        pattern=pattern,
        max_splits=max_splits,
        reverse=reverse,
        bytestring_to_string=False,
    )
    out = ak._do.recursively_apply(ak.operations.to_layout(array), action, behavior)

    return wrap_layout(out, behavior, highlevel)
