# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("from_arrow",)

np = NumpyMetadata.instance()


@high_level_function()
def from_arrow(
    array, *, generate_bitmasks=False, highlevel=True, behavior=None, attrs=None
):
    """Converts an Apache Arrow array into an Awkward Array.

    This function always preserves the values of a dataset; i.e. the Python
    objects returned by #ak.to_list are identical to the Python objects
    returned by Arrow's `to_pylist` method. If #ak.to_arrow was invoked with
    `extensionarray=True`, this function also preserves the data type
    (high-level #ak.types.Type, though not the low-level #ak.forms.Form), even
    through Parquet, making Parquet a good way to save Awkward Arrays for later
    use.

    Any attrs that #ak.to_arrow or #ak.to_arrow_table stored in the Arrow data
    are restored (as are those written by pandas, in the `"PANDAS_ATTRS"`
    schema metadata), unless overridden by the `attrs` argument.

    Because awkward uses numpy's dtype system, timestamp types do not have
    timezones. If encountering timestamp types with timezones in the input
    arrow data, they will be silently dropped.

    See also #ak.to_arrow, #ak.to_arrow_table, #ak.from_parquet, #ak.from_arrow_schema.

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
        attrs (None or dict): Custom attributes for the output array, if
            high-level. These take precedence over any attrs stored in the
            Arrow data itself.

    Returns:
        An #ak.Array built from the given Apache Arrow array.
    """
    return _impl(array, generate_bitmasks, highlevel, behavior, attrs)


def _impl(array, generate_bitmasks, highlevel, behavior, attrs):
    import awkward._connect.pyarrow

    pyarrow = awkward._connect.pyarrow.pyarrow

    stored_attrs = _stored_attrs(array)
    if stored_attrs:
        # an explicit 'attrs' argument wins over what was stored in the Arrow data
        attrs = {**stored_attrs, **(attrs or {})}

    ctx = HighLevelContext(behavior=behavior, attrs=attrs).finalize()

    out = awkward._connect.pyarrow.handle_arrow(
        array, generate_bitmasks=generate_bitmasks, pass_empty_field=True
    )

    if isinstance(array, (pyarrow.lib.Array, pyarrow.lib.ChunkedArray)):
        (
            awkwardarrow_type,
            _storage_type,
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

    return ctx.wrap(out, highlevel=highlevel)


def _stored_attrs(array) -> dict:
    """
    The attrs that #ak.to_arrow or #ak.to_arrow_table wrote into `array`: a Table
    or RecordBatch carries them in its schema metadata, whereas a bare Array or
    ChunkedArray can only carry them in its (extension) type.
    """
    import awkward._connect.pyarrow

    pyarrow = awkward._connect.pyarrow.pyarrow

    if isinstance(array, (pyarrow.lib.Table, pyarrow.lib.RecordBatch)):
        return awkward._connect.pyarrow.attrs_from_schema_metadata(
            array.schema.metadata
        )

    elif isinstance(array, (pyarrow.lib.Array, pyarrow.lib.ChunkedArray)):
        awkwardarrow_type, _ = awkward._connect.pyarrow.to_awkwardarrow_storage_types(
            array.type
        )
        if awkwardarrow_type is not None and awkwardarrow_type.attrs is not None:
            return awkwardarrow_type.attrs

    return {}
