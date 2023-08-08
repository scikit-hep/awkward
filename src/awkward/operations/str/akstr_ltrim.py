# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("ltrim",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def ltrim(array, characters, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        characters (str or bytes): Individual characters to be trimmed
            from the string.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Removes any leading characters of `characters` from any string or
    bytestring-valued data.

    If the data are strings, `characters` are interpreted as unordered,
    individual codepoints.

    If the data are bytestrings, `characters` are interpreted as unordered,
    individual bytes.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_ltrim](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_ltrim.html)
    or
    [pyarrow.compute.ascii_ltrim](https://arrow.apache.org/docs/python/generated/pyarrow.compute.ascii_ltrim.html)
    on strings and bytestrings, respectively.

    See also: #ak.str.ltrim_whitespace.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, characters, highlevel, behavior)


def _impl(array, characters, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.ltrim")
    behavior = behavior_of(array, behavior=behavior)

    out = ak._do.recursively_apply(
        ak.operations.to_layout(array),
        ak.operations.str._get_ufunc_action(
            pc.utf8_ltrim, pc.ascii_ltrim, characters, bytestring_to_string=True
        ),
        behavior,
    )

    return wrap_layout(out, behavior, highlevel)
