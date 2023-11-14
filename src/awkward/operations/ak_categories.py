# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext

__all__ = ("categories",)


@high_level_function()
def categories(array, highlevel=True, *, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    If the `array` is categorical (contains #ak.contents.IndexedArray or
    #ak.contents.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns its categories.

    See also #ak.is_categorical, #ak.str.to_categorical, #ak.from_categorical.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    output = None

    def action(layout, **kwargs):
        nonlocal output

        if layout.parameter("__array__") == "categorical":
            output = layout.content
            return layout

        else:
            return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    ak._do.recursively_apply(layout, action)

    return ctx.wrap(output, highlevel=highlevel)
