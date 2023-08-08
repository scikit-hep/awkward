# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("join",)

import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def join(array, separator, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        separator (str, bytes, or array of them to broadcast): separator to
            insert between strings. If array-like, `separator` is broadcast
            against `array`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Concatenate the strings in `array`. The `separator` is inserted between
    each string. If array-like, `separator` is broadcast against `array` which
    permits a unique separator for each list of strings in `array`.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.binary_join](https://arrow.apache.org/docs/python/generated/pyarrow.compute.binary_join.html).

    See also: #ak.str.join_element_wise.
    """
    # Dispatch
    yield (array, separator)

    # Implementation
    return _impl(array, separator, highlevel, behavior)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


def _impl(array, separator, highlevel, behavior):
    import awkward._connect.pyarrow  # noqa: F401, I001
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow

    import pyarrow.compute as pc

    def apply_unary(layout, **kwargs):
        if not (layout.is_list and _is_maybe_optional_list_of_string(layout.content)):
            return

        arrow_array = to_arrow(
            # Arrow needs an option type here
            layout.copy(content=ak.contents.UnmaskedArray.simplified(layout.content)),
            extensionarray=False,
            # This kernel requires non-large string/bytestrings
            string_to32=True,
            bytestring_to32=True,
        )
        return from_arrow(
            pc.binary_join(arrow_array, separator),
            highlevel=False,
        )

    def apply_binary(layouts, **kwargs):
        layout, separator_layout = layouts
        if not (layout.is_list and _is_maybe_optional_list_of_string(layout.content)):
            return

        if not _is_maybe_optional_list_of_string(separator_layout):
            raise TypeError(
                f"`separator` must be a list of (possibly missing) strings, not {ak.type(separator_layout)}"
            )

        # We have (maybe option/indexed type wrapping) strings
        layout_arrow = to_arrow(
            # Arrow needs an option type here
            layout.copy(content=ak.contents.UnmaskedArray.simplified(layout.content)),
            extensionarray=False,
            # This kernel requires non-large string/bytestrings
            string_to32=True,
            bytestring_to32=True,
        )
        separator_arrow = to_arrow(
            separator_layout,
            extensionarray=False,
            # This kernel requires non-large string/bytestrings
            string_to32=True,
            bytestring_to32=True,
        )
        return (
            from_arrow(
                pc.binary_join(layout_arrow, separator_arrow),
                highlevel=False,
            ),
        )

    layout = ak.to_layout(array, allow_record=False, allow_other=True)
    behavior = behavior_of(array, separator, behavior=behavior)
    if isinstance(separator, (bytes, str)):
        out = ak._do.recursively_apply(layout, apply_unary, behavior=behavior)
    else:
        separator_layout = ak.to_layout(separator, allow_record=False, allow_other=True)
        (out,) = ak._broadcasting.broadcast_and_apply(
            (layout, separator_layout), apply_binary, behavior
        )

    return wrap_layout(out, highlevel=highlevel, behavior=behavior)
