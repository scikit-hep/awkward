# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("is_categorical",)


@high_level_function()
def is_categorical(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    If the `array` is categorical (contains #ak.contents.IndexedArray or
    #ak.contents.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns True;
    otherwise, it returns False.

    See also #ak.categories, #ak.str.to_categorical, #ak.from_categorical.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    layout = ak.operations.to_layout(
        array, allow_record=False, primitive_policy="error"
    )
    return layout.purelist_parameter("__array__") == "categorical"
