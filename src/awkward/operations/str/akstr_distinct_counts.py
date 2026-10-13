# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Distinct string values and counts for each final string-list axis."""

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("distinct_counts",)


def _is_string(layout):
    return layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}


def _is_maybe_optional_string(layout):
    if _is_string(layout):
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_string(layout.content)
    else:
        return False


def _is_list_of_strings(layout):
    return layout.is_list and _is_maybe_optional_string(layout.content)


def _value_counts_per_sublist(list_array):
    from awkward._connect.pyarrow import import_pyarrow, import_pyarrow_compute

    pa = import_pyarrow("ak.str.distinct_counts")
    pc = import_pyarrow_compute("ak.str.distinct_counts")

    value_type = list_array.type.value_type
    value_counts_type = pa.struct(
        [
            pa.field("values", value_type),
            pa.field("counts", pa.int64()),
        ]
    )
    offsets = list_array.offsets
    offset_values = offsets.to_numpy(zero_copy_only=False)
    values = list_array.values
    valid = list_array.is_valid()

    out_offsets = [0]
    out_values = []
    out_mask = []

    # PyArrow has no segmented value_counts kernel. We keep the loop at the
    # Arrow slice level and avoid materializing sublists through Python objects.
    for i in range(len(list_array)):
        if not valid[i].as_py():
            out_offsets.append(out_offsets[-1])
            out_mask.append(True)
            continue

        start = offset_values[i]
        stop = offset_values[i + 1]
        value_counts = pc.value_counts(pc.drop_null(values.slice(start, stop - start)))
        out_values.append(value_counts)
        out_offsets.append(out_offsets[-1] + len(value_counts))
        out_mask.append(False)

    values_out = (
        pa.concat_arrays(out_values)
        if out_values
        else pa.array([], type=value_counts_type)
    )
    offsets_out = pa.array(out_offsets, type=offsets.type)
    mask_out = pa.array(out_mask) if any(out_mask) else None

    if pa.types.is_large_list(list_array.type):
        return pa.LargeListArray.from_arrays(offsets_out, values_out, mask=mask_out)
    else:
        return pa.ListArray.from_arrays(offsets_out, values_out, mask=mask_out)


def _action(layout, lateral_context, **kwargs):
    from awkward.operations.str import _apply_through_arrow

    if _is_list_of_strings(layout):
        return _apply_through_arrow(
            _value_counts_per_sublist, layout, expect_option_type=False
        )

    if _is_maybe_optional_string(layout):
        return _apply_through_arrow(
            lateral_context["kernel"], layout, expect_option_type=True
        )


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

    Returns records containing each distinct value (`"values"`) and its
    frequency (`"counts"`) in an array of strings or bytestrings, applying the
    operation independently at the final string-list axis and preserving outer
    nesting.

    If `array` contains no string or bytestring data, this function returns it
    unchanged.

    Requires the pyarrow library and calls
    [pyarrow.compute.value_counts](https://arrow.apache.org/docs/python/generated/pyarrow.compute.value_counts.html).

    Traversal strategy:
      `ak.transform` walks the Awkward layout tree and calls `_action` at each
      node. When `_action` finds a list whose content is a string or bytestring
      layout, it applies the Arrow kernel to slices defined by that list's
      offsets and returns a replacement layout, which stops recursion into that
      subtree. Other nodes return None, allowing `ak.transform` to continue.
      This follows Awkward's public traversal API and uses `_apply_through_arrow`
      for Arrow conversion rather than manually rebuilding the surrounding
      nesting with `recursively_apply`.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.distinct_counts")
    lateral_context = {"kernel": pc.value_counts}

    return ak.transform(
        _action,
        array,
        lateral_context=lateral_context,
        return_value="original",
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
