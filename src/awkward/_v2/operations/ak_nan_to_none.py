# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def nan_to_none(array, highlevel=True, behavior=None):

    """
    Args:
        array: Array whose `NaN` values should be converted to None (missing values).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts NaN ("not a number") into None, i.e. missing values with option-type.

    See also #ak.nan_to_num to convert NaN or infinity to specified values.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.nan_to_none",
        dict(
            array=array,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    def action(layout, continuation, **kwargs):
        if isinstance(layout, ak._v2.contents.NumpyArray) and issubclass(
            layout.dtype.type, np.floating
        ):
            return ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(
                    layout.nplike.isnan(layout.data), nplike=layout.nplike
                ),
                layout,
                valid_when=False,
            )

        elif (layout.is_OptionType or layout.is_IndexedType) and (
            isinstance(layout.content, ak._v2.contents.NumpyArray)
            and issubclass(layout.content.dtype.type, np.floating)
        ):
            return continuation().simplify_optiontype()

        else:
            return None

    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    out = layout.recursively_apply(action, behavior)
    return ak._v2._util.wrap(out, behavior, highlevel)
