# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend

__all__ = ("join",)

typetracer = TypeTracerBackend.instance()


@high_level_function(module="ak.str")
def join(array, separator, *, highlevel=True, behavior=None, attrs=None):
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
        attrs (None or dict): Custom attributes for the output array, if
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
    yield array, separator

    # Implementation
    return _impl(array, separator, highlevel, behavior, attrs)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


def _impl(array, separator, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.join")

    from awkward.operations.str import _apply_through_arrow

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout, maybe_separator_layout = ensure_same_backend(
            ctx.unwrap(array, allow_record=False, primitive_policy="error"),
            ctx.unwrap(
                separator,
                allow_record=False,
                allow_unknown=False,
                primitive_policy="error",
                string_policy="pass-through",
            ),
        )

    if isinstance(maybe_separator_layout, (bytes, str)):

        def apply_unary(layout, **kwargs):
            if not (
                layout.is_list and _is_maybe_optional_list_of_string(layout.content)
            ):
                return

            return _apply_through_arrow(
                pc.binary_join,
                # Arrow needs an option type here
                layout.copy(
                    content=ak.contents.UnmaskedArray.simplified(layout.content)
                ),
                maybe_separator_layout,
                # This kernel requires non-large string/bytestrings
                string_to32=True,
                bytestring_to32=True,
            )

        out = ak._do.recursively_apply(layout, apply_unary)
    else:

        def apply_binary(layouts, **kwargs):
            if not (
                layouts[0].is_list
                and _is_maybe_optional_list_of_string(layouts[0].content)
            ):
                return

            if not _is_maybe_optional_list_of_string(layouts[1]):
                raise TypeError(
                    f"`separator` must be a list of (possibly missing) strings, not {ak.type(layouts[1])}"
                )

            return (
                _apply_through_arrow(
                    pc.binary_join,
                    # Arrow needs an option type here
                    layouts[0].copy(
                        content=ak.contents.UnmaskedArray.simplified(layouts[0].content)
                    ),
                    layouts[1],
                    # This kernel requires non-large string/bytestrings
                    string_to32=True,
                    bytestring_to32=True,
                ),
            )

        (out,) = ak._broadcasting.broadcast_and_apply(
            (layout, maybe_separator_layout), apply_binary
        )

    return ctx.wrap(out, highlevel=highlevel)
