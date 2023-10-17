# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("is_numeric",)


@high_level_function(module="ak.str")
def is_numeric(array, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Replaces any string-valued data with True if the string is non-empty and
    consists only of numeric Unicode characters, False otherwise.

    Replaces any bytestring-valued data with True if the string is non-empty
    and consists only of numeric Unicode characters, False otherwise.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_is_numeric](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_is_numeric.html)
    or
    [pyarrow.compute.utf8_is_numeric](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_is_numeric.html)
    on strings and bytestrings, respectively.

    (Arrow's compute module does not have an `ascii_is_numeric`.)
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("c")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = ak._do.recursively_apply(
        layout,
        ak.operations.str._get_ufunc_action(
            pc.utf8_is_numeric, pc.utf8_is_numeric, bytestring_to_string=True
        ),
    )

    return ctx.wrap(out, highlevel=highlevel)
