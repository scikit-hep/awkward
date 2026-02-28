# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._do import recursively_apply
from awkward._layout import HighLevelContext

__all__ = ("uniques",)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


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

    Returns one copy of each distinct value in a one-dimensional array of
    strings or bytestrings.

    If `array` contains no string or bytestring data, this function returns it
    unchanged.

    Requires the pyarrow library and calls
    [pyarrow.compute.unique](https://arrow.apache.org/docs/python/generated/pyarrow.compute.unique.html).
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow, import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pa = import_pyarrow("ak.str.uniques")
    pc = import_pyarrow_compute("ak.str.uniques")

    def unique_per_sublist(list_array):
        value_type = list_array.type.value_type
        output = []
        for sublist in list_array.to_pylist():
            if sublist is None:
                output.append(None)
            else:
                subarray = pa.array(sublist, type=value_type)
                output.append(pc.unique(subarray).to_pylist())
        return pa.array(output, type=pa.large_list(value_type))

    def unique_whole_array(array_1d):
        return pc.unique(array_1d)

    any_list_of_strings = False

    def action(layout, **kwargs):
        nonlocal any_list_of_strings
        if layout.is_list and _is_maybe_optional_list_of_string(layout.content):
            any_list_of_strings = True
            return _apply_through_arrow(
                unique_per_sublist, layout, expect_option_type=False
            )
        elif _is_maybe_optional_list_of_string(layout):
            any_list_of_strings = True
            return _apply_through_arrow(
                unique_whole_array, layout, expect_option_type=False
            )

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = recursively_apply(layout, action)
    if not any_list_of_strings:
        out = layout

    return ctx.wrap(out, highlevel=highlevel)
