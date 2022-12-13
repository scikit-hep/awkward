# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_arrow(
    array,
    *,
    list_to32=False,
    string_to32=False,
    bytestring_to32=False,
    emptyarray_to=None,
    categorical_as_dictionary=False,
    extensionarray=True,
    count_nulls=True,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
            if they're small enough, even if it means an extra conversion. Otherwise,
            signed 32-bit #ak.types.ListType maps to Arrow `ListType`,
            signed 64-bit #ak.types.ListType maps to Arrow `LargeListType`,
            and unsigned 32-bit #ak.types.ListType picks whichever Arrow type its
            values fit into.
        string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
        bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
        emptyarray_to (None or dtype): If None, #ak.types.UnknownType maps to Arrow's
            null type; otherwise, it is converted a given numeric dtype.
        categorical_as_dictionary (bool): If True, #ak.contents.IndexedArray and
            #ak.contents.IndexedOptionArray labeled with `__array__ = "categorical"`
            are mapped to Arrow `DictionaryArray`; otherwise, the projection is
            evaluated before conversion (always the case without
            `__array__ = "categorical"`).
        extensionarray (bool): If True, this function returns extended Arrow arrays
            (at all levels of nesting), which preserve metadata so that Awkward \u2192
            Arrow \u2192 Awkward preserves the array's #ak.types.Type (though not
            the #ak.forms.Form). If False, this function returns generic Arrow arrays
            that might be needed for third-party tools that don't recognize Arrow's
            extensions. Even with `extensionarray=False`, the values produced by
            Arrow's `to_pylist` method are the same as the values produced by Awkward's
            #ak.to_list.
        count_nulls (bool): If True, count the number of missing values at each level
            and include these in the resulting Arrow array, which makes some downstream
            applications faster. If False, skip the up-front cost of counting them.

    Converts an Awkward Array into an Apache Arrow array.

    This produces arrays of type `pyarrow.Array`. You might need to further
    manipulations (using the pyarrow library) to build a `pyarrow.ChunkedArray`,
    a `pyarrow.RecordBatch`, or a `pyarrow.Table`. For the latter, see #ak.to_arrow_table.

    This function always preserves the values of a dataset; i.e. the Python objects
    returned by #ak.to_list are identical to the Python objects returned by Arrow's
    `to_pylist` method. With `extensionarray=True`, this function also preserves the
    data type (high-level #ak.types.Type, though not the low-level #ak.forms.Form),
    even through Parquet, making Parquet a good way to save Awkward Arrays for later
    use. If any third-party tools don't recognize Arrow's extension arrays, set this
    option to False for plain Arrow arrays.

    See also #ak.from_arrow, #ak.to_arrow_table, #ak.to_parquet, #ak.from_arrow_schema.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_arrow",
        dict(
            array=array,
            list_to32=list_to32,
            string_to32=string_to32,
            bytestring_to32=bytestring_to32,
            emptyarray_to=emptyarray_to,
            categorical_as_dictionary=categorical_as_dictionary,
            extensionarray=extensionarray,
            count_nulls=count_nulls,
        ),
    ):
        return _impl(
            array,
            list_to32,
            string_to32,
            bytestring_to32,
            emptyarray_to,
            categorical_as_dictionary,
            extensionarray,
            count_nulls,
        )


def _impl(
    array,
    list_to32,
    string_to32,
    bytestring_to32,
    emptyarray_to,
    categorical_as_dictionary,
    extensionarray,
    count_nulls,
):
    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)
    if isinstance(layout, ak.record.Record):
        layout = layout.array[layout.at : layout.at + 1]
        record_is_scalar = True
    else:
        record_is_scalar = False

    return layout.to_arrow(
        list_to32=list_to32,
        string_to32=string_to32,
        bytestring_to32=bytestring_to32,
        emptyarray_to=emptyarray_to,
        categorical_as_dictionary=categorical_as_dictionary,
        extensionarray=extensionarray,
        count_nulls=count_nulls,
        record_is_scalar=record_is_scalar,
    )
