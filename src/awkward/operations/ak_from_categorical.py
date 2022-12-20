# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_categorical(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This function replaces categorical data with non-categorical data (by
    removing the label that declares it as such).

    This is a metadata-only operation; the running time does not scale with the
    size of the dataset. (Conversion to categorical is expensive; conversion
    from categorical is cheap.)

    See also #ak.is_categorical, #ak.categories, #ak.to_categorical,
    #ak.from_categorical.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_categorical",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    def action(layout, **kwargs):
        if layout.parameter("__array__") == "categorical":
            out = ak.operations.with_parameter(
                layout, "__array__", None, highlevel=False
            )
            return out

        else:
            return None

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    out = ak._do.recursively_apply(layout, action, behavior=behavior)
    if highlevel:
        return ak._util.wrap(out, behavior)
    else:
        return out
