# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("categories",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function()
def categories(array, highlevel=True):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.

    If the `array` is categorical (contains #ak.contents.IndexedArray or
    #ak.contents.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns its categories.

    See also #ak.is_categorical, #ak.to_categorical, #ak.from_categorical.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel)


def _impl(array, highlevel):
    output = [None]

    def action(layout, **kwargs):
        if layout.parameter("__array__") == "categorical":
            output[0] = layout.content
            return layout

        else:
            return None

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = behavior_of(array)
    ak._do.recursively_apply(layout, action, behavior=behavior)

    return wrap_layout(output[0], behavior, highlevel)
