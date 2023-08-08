# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("is_in",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def is_in(array, value_set, *, skip_nones=False, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        value_set: Array-like data (anything #ak.to_layout recognizes), set of
            values to search for in `array`.
        skip_nones (bool): If True, None values in `array` are not matched
            against `value_set`; otherwise, None is considered a legal value.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns True for each string in `array` if it matches any pattern in
    `value_set`; otherwise, returns False.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.is_in](https://arrow.apache.org/docs/python/generated/pyarrow.compute.is_in.html).
    """
    # Dispatch
    yield (array, value_set)

    # Implementation
    return _impl(array, value_set, skip_nones, highlevel, behavior)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


def _impl(array, value_set, skip_nones, highlevel, behavior):
    import awkward._connect.pyarrow  # noqa: F401, I001

    import pyarrow.compute as pc

    layout = ak.to_layout(array, allow_record=False, allow_other=True)
    value_set_layout = ak.to_layout(value_set, allow_record=False, allow_other=True)

    if not _is_maybe_optional_list_of_string(value_set_layout):
        raise TypeError("`value_set` must be 1D array of (possibly missing) strings")

    behavior = behavior_of(array, value_set, behavior=behavior)

    def apply(layout, **kwargs):
        if _is_maybe_optional_list_of_string(layout):
            return ak.from_arrow(
                pc.is_in(
                    ak.to_arrow(layout, extensionarray=False),
                    ak.to_arrow(value_set_layout, extensionarray=False),
                    skip_nulls=skip_nones,
                ),
                highlevel=False,
            )

    out = ak._do.recursively_apply(layout, apply, behavior=behavior)

    return wrap_layout(out, highlevel=highlevel, behavior=behavior)
