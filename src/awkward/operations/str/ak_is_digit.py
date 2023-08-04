# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("is_digit",)

import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function
def is_digit(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Replaces any string-valued data True iff the string is non-empty and consists only of Unicode digits.

    Replaces any bytestring-valued data True iff the string is non-empty and consists only of Unicode digits.

    Note: this function does not raise an error if the `array` does
    not contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.utf8_isalpha](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_is_digit.html)
    or
    [pyarrow.compute.utf8_isalpha](https://arrow.apache.org/docs/python/generated/pyarrow.compute.utf8_is_digit.html)
    on strings and bytestrings, respectively.

    (Arrow's compute module does not have an `ascii_is_digit`.)
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    import awkward._connect.pyarrow  # noqa: F401, I001

    import pyarrow.compute as pc

    behavior = behavior_of(array, behavior=behavior)

    out = ak._do.recursively_apply(
        ak.operations.to_layout(array),
        ak.operations.str._get_action(
            pc.utf8_is_digit, pc.utf8_is_digit, bytestring_to_string=True
        ),
        behavior,
    )

    return wrap_layout(out, behavior, highlevel)