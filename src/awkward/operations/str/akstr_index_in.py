# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend

__all__ = ("index_in",)

typetracer = TypeTracerBackend.instance()


@high_level_function(module="ak.str")
def index_in(
    array, value_set, *, skip_nones=False, highlevel=True, behavior=None, attrs=None
):
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
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns the index of the first pattern in `value_set` that each string in
    `array` matches. If the string is not found within `value_set`, then the
    index is set to None.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.index_in](https://arrow.apache.org/docs/python/generated/pyarrow.compute.index_in.html).
    """
    # Dispatch
    yield (array, value_set)

    # Implementation
    return _impl(array, value_set, skip_nones, highlevel, behavior, attrs)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


def _impl(array, value_set, skip_nones, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute
    from awkward.operations.str import _apply_through_arrow

    pc = import_pyarrow_compute("ak.str.index_in")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout, value_set_layout = ensure_same_backend(
            ctx.unwrap(array, allow_record=False),
            ctx.unwrap(value_set, allow_record=False),
        )

    if not _is_maybe_optional_list_of_string(value_set_layout):
        raise TypeError("`value_set` must be 1D array of (possibly missing) strings")

    def apply(layout, **kwargs):
        if _is_maybe_optional_list_of_string(layout):
            return _apply_through_arrow(
                pc.index_in,
                layout,
                value_set_layout,
                skip_nulls=skip_nones,
                expect_option_type=True,
                generate_bitmasks=True,
            )

    out = ak._do.recursively_apply(layout, apply)

    return ctx.wrap(out, highlevel=highlevel)
