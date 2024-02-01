# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("rpad",)


@high_level_function(module="ak.str")
def rpad(array, width, padding=" ", *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        width (int): Desired string length.
        padding (str or bytes): What to pad the string with. Should be one
            codepoint or byte.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Replaces any string or bytestring-valued data with left-aligned
    strings/bytestrings of a given `width`, padding the right side with the
    given `padding` codepoint or byte.

    If the data are strings, `width` is measured in codepoints and `padding`
    must be one codepoint.

    If the data are bytestrings, `width` is measured in bytes and `padding`
    must be one byte.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_rpad](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_rpad.html)
    or
    [pyarrow.compute.ascii_rpad](https://arrow.apache.org/docs/python/generated/pyarrow.compute.ascii_rpad.html)
    on strings and bytestrings, respectively.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, width, padding, highlevel, behavior, attrs)


def _impl(array, width, padding, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("d")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = ak._do.recursively_apply(
        layout,
        ak.operations.str._get_ufunc_action(
            pc.utf8_rpad, pc.ascii_rpad, width, padding, bytestring_to_string=True
        ),
    )

    return ctx.wrap(out, highlevel=highlevel)
