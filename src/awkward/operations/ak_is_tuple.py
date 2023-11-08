# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("is_tuple",)


@high_level_function()
def is_tuple(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    If `array` is a record, this returns True if the record is a tuple.
    If `array` is an array, this returns True if the outermost record is a tuple.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    layout = ak.operations.ak_to_layout._impl(
        array,
        allow_record=True,
        allow_unknown=False,
        none_policy="error",
        regulararray=True,
        use_from_iter=True,
        primitive_policy="error",
        string_policy="as-characters",
    )

    return layout.is_tuple
