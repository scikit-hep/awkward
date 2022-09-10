# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_categorical(array, highlevel=True):
    """
    Args:
        array: Awkward Array from which to remove the 'categorical' parameter.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    This function replaces categorical data with non-categorical data (by
    removing the label that declares it as such).

    This is a metadata-only operation; the running time does not scale with the
    size of the dataset. (Conversion to categorical is expensive; conversion
    from categorical is cheap.)

    See also #ak.is_categorical, #ak.categories, #ak.to_categorical,
    #ak.from_categorical.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_categorical",
        dict(array=array, highlevel=highlevel),
    ):
        return _impl(array, highlevel)


def _impl(array, highlevel):
    def action(layout, **kwargs):
        if layout.parameter("__array__") == "categorical":
            out = ak._v2.operations.with_parameter(
                layout, "__array__", None, highlevel=False
            )
            return out

        else:
            return None

    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    out = layout.recursively_apply(action)
    if highlevel:
        return ak._v2._util.wrap(out, ak._v2._util.behavior_of(array))
    else:
        return out
