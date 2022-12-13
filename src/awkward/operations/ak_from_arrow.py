# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def from_arrow(array, *, generate_bitmasks=False, highlevel=True, behavior=None):
    """
    Args:
        array (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`, or `pyarrow.Table`):
            Apache Arrow array to convert into an  Awkward Array.
        generate_bitmasks (bool): If enabled and Arrow/Parquet does not have Awkward
            metadata, `generate_bitmasks=True` creates empty bitmasks for nullable
            types that don't have bitmasks in the Arrow/Parquet data, so that the
            Form (BitMaskedForm vs UnmaskedForm) is predictable.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an Apache Arrow array into an Awkward Array.

    This function always preserves the values of a dataset; i.e. the Python objects
    returned by #ak.to_list are identical to the Python objects returned by Arrow's
    `to_pylist` method. If #ak.to_arrow was invoked with `extensionarray=True`, this
    function also preserves the data type (high-level #ak.types.Type, though not the
    low-level #ak.forms.Form), even through Parquet, making Parquet a good way to save
    Awkward Arrays for later use.

    See also #ak.to_arrow, #ak.to_arrow_table, #ak.from_parquet, #ak.from_arrow_schema.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_arrow",
        dict(
            array=array,
            generate_bitmasks=generate_bitmasks,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, generate_bitmasks, highlevel, behavior)


def _impl(array, generate_bitmasks, highlevel, behavior):
    import awkward._connect.pyarrow

    pyarrow = awkward._connect.pyarrow.pyarrow

    out = awkward._connect.pyarrow.handle_arrow(
        array, generate_bitmasks=generate_bitmasks, pass_empty_field=True
    )

    if isinstance(array, (pyarrow.lib.Array, pyarrow.lib.ChunkedArray)):
        (
            awkwardarrow_type,
            storage_type,
        ) = awkward._connect.pyarrow.to_awkwardarrow_storage_types(array.type)

        if awkwardarrow_type is None:
            if isinstance(out, ak.contents.UnmaskedArray):
                out = awkward._connect.pyarrow.remove_optiontype(out)
        else:
            if awkwardarrow_type.mask_type in (None, "IndexedArray"):
                out = awkward._connect.pyarrow.remove_optiontype(out)

            if awkwardarrow_type.record_is_scalar:
                out = out._getitem_at(0)

    def remove_revertable(layout, **kwargs):
        if hasattr(layout, "__pyarrow_original"):
            del layout.__pyarrow_original

    ak._do.recursively_apply(out, remove_revertable)

    return ak._util.wrap(out, behavior, highlevel)
