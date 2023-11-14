# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("extract_regex",)


@high_level_function(module="ak.str")
def extract_regex(array, pattern, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Regular expression with named capture fields.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns None for every string in `array` if it does not match `pattern`;
    otherwise, a record whose fields are named capture groups and whose
    contents are the substrings they've captured.

    Uses [Google RE2](https://github.com/google/re2/wiki/Syntax), and `pattern` must
    contain named groups. The syntax for a named group is `(?P<...>...)` in which
    the first `...` is a name and the last `...` is a regular expression.

    For example,

        >>> array = ak.Array([["one1", "two2", "three3"], [], ["four4", "five5"]])
        >>> result = ak.str.extract_regex(array, "(?P<vowel>[aeiou])(?P<number>[0-9]+)")
        >>> result.show(type=True)
        type: 3 * var * ?{
            vowel: ?string,
            number: ?string
        }
        [[{vowel: 'e', number: '1'}, {vowel: 'o', number: '2'}, {vowel: 'e', number: '3'}],
         [],
         [None, {vowel: 'e', number: '5'}]]

    (The string `"four4"` does not match because the vowel is not immediately before
    the number.)

    Regular expressions with unnamed groups or features not implemented by RE2 raise an error.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.extract_regex](https://arrow.apache.org/docs/python/generated/pyarrow.compute.extract_regex.html).
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, pattern, highlevel, behavior, attrs)


def _impl(array, pattern, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("x")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array)

    out = ak._do.recursively_apply(
        layout,
        ak.operations.str._get_ufunc_action(
            pc.extract_regex,
            pc.extract_regex,
            pattern,
            generate_bitmasks=True,
            bytestring_to_string=False,
            expect_option_type=True,
        ),
    )

    return ctx.wrap(out, highlevel=highlevel)
