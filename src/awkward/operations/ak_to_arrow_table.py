# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_arrow_table",)
import json

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


@high_level_function()
def to_arrow_table(
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
    # Dispatch
    yield (array,)

    # Implementation
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
    from awkward._connect.pyarrow import direct_Content_subclass, pyarrow

    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)
    if isinstance(layout, ak.record.Record):
        layout = layout.array[layout.at : layout.at + 1]
        record_is_scalar = True
    else:
        record_is_scalar = False

    wrapper_layouts = [layout]
    while wrapper_layouts[-1].is_option or wrapper_layouts[-1].is_indexed:
        wrapper_layouts.append(wrapper_layouts[-1].content)

    parameters = None
    paarrays, pafields = [], []
    if wrapper_layouts[-1].is_record and not wrapper_layouts[-1].is_tuple:
        record_content = wrapper_layouts[-1]
        optiontype_fields = []
        for name in record_content.fields:
            outer_field_content = layout[name]
            paarrays.append(
                outer_field_content.to_arrow(
                    list_to32=list_to32,
                    string_to32=string_to32,
                    bytestring_to32=bytestring_to32,
                    emptyarray_to=emptyarray_to,
                    categorical_as_dictionary=categorical_as_dictionary,
                    extensionarray=extensionarray,
                    count_nulls=count_nulls,
                    record_is_scalar=record_is_scalar,
                )
            )
            pafields.append(
                pyarrow.field(name, paarrays[-1].type).with_nullable(
                    outer_field_content.is_option
                )
            )
            if record_content.contents[record_content.field_to_index(name)].is_option:
                optiontype_fields.append(name)

        parameters = [
            {"optiontype_fields": optiontype_fields},
            {"record_is_scalar": record_is_scalar},
        ]

        # We build a table from the _contents_ of the record layout. Therefore,
        # we must explicitly include the parameters of each layout above and including the record
        for x in wrapper_layouts:
            parameters.append({direct_Content_subclass(x).__name__: x._parameters})

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
                record_is_scalar=record_is_scalar,
            )
        )
        pafields.append(
            pyarrow.field("", paarrays[-1].type).with_nullable(
                layout.is_option or layout.is_identity_like
            )
        )

    batch = pyarrow.RecordBatch.from_arrays(paarrays, schema=pyarrow.schema(pafields))
    out = pyarrow.Table.from_batches([batch])

    if parameters is None:
        return out
    else:
        return out.replace_schema_metadata({"ak:parameters": json.dumps(parameters)})
