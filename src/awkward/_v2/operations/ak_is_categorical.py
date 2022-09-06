# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def is_categorical(array):
    """
    Args:
        array: A possibly-categorical Awkward Array.

    If the `array` is categorical (contains #ak.layout.IndexedArray or
    #ak.layout.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns True;
    otherwise, it returns False.

    See also #ak.categories, #ak.to_categorical, #ak.from_categorical.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.is_categorical",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    return layout.purelist_parameter("__array__") == "categorical"
