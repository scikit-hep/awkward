# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_arrow_table(
    array,
    list_to32=False,
    string_to32=False,
    bytestring_to32=False,
    emptyarray_to=None,
    categorical_as_dictionary=False,
    extensionarray=True,
    count_nulls=True,
):
    u"""
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
        categorical_as_dictionary (bool): If True, #ak.layout.IndexedArray and
            #ak.layout.IndexedOptionArray labeled with `__array__ = "categorical"`
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

    Converts an Awkward Array into an Apache Arrow table.

    This produces arrays of type `pyarrow.Table`. If you want an Arrow array,
    see #ak.to_arrow.

    This function always preserves the values of a dataset; i.e. the Python objects
    returned by #ak.to_list are identical to the Python objects returned by Arrow's
    `to_pylist` method. With `extensionarray=True`, this function also preserves the
    data type (high-level #ak.types.Type, though not the low-level #ak.forms.Form),
    even through Parquet, making Parquet a good way to save Awkward Arrays for later
    use. If any third-party tools don't recognize Arrow's extension arrays, set this
    option to False for plain Arrow arrays.

    See also #ak.from_arrow, #ak.to_arrow, #ak.to_parquet.
    """
    from awkward._v2._connect.pyarrow import pyarrow

    layout = ak._v2.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    check = [layout]
    while check[-1].is_OptionType or check[-1].is_IndexedType:
        check.append(check[-1].content)

    parameters = None
    paarrays, pafields = [], []
    if check[-1].is_RecordType and not check[-1].is_tuple:
        for name in check[-1].fields:
            paarrays.append(
                layout[name].to_arrow(
                    list_to32=list_to32,
                    string_to32=string_to32,
                    bytestring_to32=bytestring_to32,
                    emptyarray_to=emptyarray_to,
                    categorical_as_dictionary=categorical_as_dictionary,
                    extensionarray=extensionarray,
                    count_nulls=count_nulls,
                )
            )
            pafields.append(
                pyarrow.field(name, paarrays[-1].type).with_nullable(
                    layout.is_OptionType
                )
            )
        parameters = []
        for x in check:
            parameters.append(
                {ak._v2._util.direct_Content_subclass(x).__name__: x._parameters}
            )

    else:
        paarrays.append(
            layout.to_arrow(
                list_to32=list_to32,
                string_to32=string_to32,
                bytestring_to32=bytestring_to32,
                emptyarray_to=emptyarray_to,
                categorical_as_dictionary=categorical_as_dictionary,
                extensionarray=extensionarray,
                count_nulls=count_nulls,
            )
        )
        pafields.append(
            pyarrow.field("", paarrays[-1].type).with_nullable(layout.is_OptionType)
        )

    batch = pyarrow.RecordBatch.from_arrays(paarrays, schema=pyarrow.schema(pafields))
    out = pyarrow.Table.from_batches([batch])

    if parameters is None:
        return out
    else:
        return out.replace_schema_metadata({"ak:parameters": json.dumps(parameters)})
