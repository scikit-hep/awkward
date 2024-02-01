# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("trim",)


@high_level_function(module="ak.str")
def trim(array, characters, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        characters (str or bytes): Individual characters to be trimmed from
            the string.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Removes any leading or trailing characters of `characters` from any string
    or bytestring-valued data.

    If the data are strings, `characters` are interpreted as unordered,
    individual codepoints.

    If the data are bytestrings, `characters` are interpreted as unordered,
    individual bytes.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_trim](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_trim.html)
    or
    [pyarrow.compute.ascii_trim](https://arrow.apache.org/docs/python/generated/pyarrow.compute.ascii_trim.html)
    on strings and bytestrings, respectively.

    See also: #ak.str.trim_whitespace.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, characters, highlevel, behavior, attrs)


def _impl(array, characters, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("m")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = ak._do.recursively_apply(
        layout,
        ak.operations.str._get_ufunc_action(
            pc.utf8_trim, pc.ascii_trim, characters, bytestring_to_string=True
        ),
    )

    return ctx.wrap(out, highlevel=highlevel)
