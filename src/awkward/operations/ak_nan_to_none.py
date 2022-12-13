# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def nan_to_none(array, *, highlevel=True, behavior=None):

    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts NaN ("not a number") into None, i.e. missing values with option-type.

    See also #ak.nan_to_num to convert NaN or infinity to specified values.
    """
    with ak._errors.OperationErrorContext(
        "ak.nan_to_none",
        dict(
            array=array,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    def action(layout, continuation, **kwargs):
        if isinstance(layout, ak.contents.NumpyArray) and issubclass(
            layout.dtype.type, np.floating
        ):
            mask = layout.backend.nplike.isnan(layout.data)
            return ak.contents.ByteMaskedArray(
                ak.index.Index8(mask, nplike=layout.backend.index_nplike),
                layout,
                valid_when=False,
            )

        elif (layout.is_option or layout.is_indexed) and (
            isinstance(layout.content, ak.contents.NumpyArray)
            and issubclass(layout.content.dtype.type, np.floating)
        ):
            return continuation()

        else:
            return None

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    out = ak._do.recursively_apply(layout, action, behavior)
    return ak._util.wrap(out, behavior, highlevel)
