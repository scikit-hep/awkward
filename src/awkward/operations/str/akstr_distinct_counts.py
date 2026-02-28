# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._do import recursively_apply
from awkward._layout import HighLevelContext

__all__ = ("distinct_counts",)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


@high_level_function(module="ak.str")
def distinct_counts(array, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns one record per distinct value in a one-dimensional array of
    strings or bytestrings. Each record contains the distinct value (`"values"`)
    and its frequency (`"counts"`).

    If `array` contains no string or bytestring data, this function returns it
    unchanged.

    Requires the pyarrow library and calls
    [pyarrow.compute.value_counts](https://arrow.apache.org/docs/python/generated/pyarrow.compute.value_counts.html).
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow, import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pa = import_pyarrow("ak.str.distinct_counts")
    pc = import_pyarrow_compute("ak.str.distinct_counts")

    def value_counts_per_sublist(list_array):
        value_type = list_array.type.value_type
        value_counts_type = pa.struct(
            [
                pa.field("values", value_type),
                pa.field("counts", pa.int64()),
            ]
        )
        output = []
        for sublist in list_array.to_pylist():
            if sublist is None:
                output.append(None)
            else:
                subarray = pa.array(sublist, type=value_type)
                output.append(pc.value_counts(subarray).to_pylist())
        return pa.array(output, type=pa.large_list(value_counts_type))

    def value_counts_whole_array(array_1d):
        return pc.value_counts(array_1d)

    any_list_of_strings = False

    def action(layout, **kwargs):
        nonlocal any_list_of_strings
        if layout.is_list and _is_maybe_optional_list_of_string(layout.content):
            any_list_of_strings = True
            return _apply_through_arrow(
                value_counts_per_sublist, layout, expect_option_type=False
            )
        elif _is_maybe_optional_list_of_string(layout):
            any_list_of_strings = True
            return _apply_through_arrow(
                value_counts_whole_array, layout, expect_option_type=False
            )

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = recursively_apply(layout, action)
    if not any_list_of_strings:
        out = layout

    return ctx.wrap(out, highlevel=highlevel)
