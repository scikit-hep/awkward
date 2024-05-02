# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any

__all__ = ("to_arrow_table",)

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

    layout = ak.operations.to_layout(array, allow_record=True, primitive_policy="error")
    if isinstance(layout, ak.record.Record):
        layout = layout.array[layout.at : layout.at + 1]
        record_is_scalar = True
    else:
        record_is_scalar = False

    # If we're building a table from an array of records, we need to keep track
    # of the "wrapper" layouts above the record
    wrapper_layouts = [layout]
    while wrapper_layouts[-1].is_option or wrapper_layouts[-1].is_indexed:
        wrapper_layouts.append(wrapper_layouts[-1].content)

    schema_parameters: list[dict[str, Any]] | None = None
    arrow_arrays, arrow_fields = [], []
    if wrapper_layouts[-1].is_record and not wrapper_layouts[-1].is_tuple:
        record_content = wrapper_layouts[-1]
        optiontype_fields: list[str] = []
        for name in record_content.fields:
            outer_field_content = layout[name]
            arrow_arrays.append(
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
            # The nullability of this arrow field is determined by whether
            # the field is considered an option w.r.t to the root, i.e.
            # accounting for options above the record layout.
            arrow_fields.append(
                pyarrow.field(name, arrow_arrays[-1].type).with_nullable(
                    outer_field_content._arrow_needs_option_type()
                )
            )

            # Is this field layout directly an option-type?
            record_field_content = record_content.contents[
                record_content.field_to_index(name)
            ]
            if record_field_content.is_option:
                optiontype_fields.append(name)

        schema_parameters = [
            {"optiontype_fields": optiontype_fields},
            {"record_is_scalar": record_is_scalar},
        ]

        # The parameters that _may_ be serialised in the Arrow schema
        # (depending upon the value of `extensionarray`) will not include those
        # that are defined on nodes above the record; slicing a RecordArray
        # drops parameters above the record. As such, we need to explicitly
        # track these in order to rebuild the layouts above the record.
        for content in wrapper_layouts:
            schema_parameters.append(
                {direct_Content_subclass(content).__name__: content._parameters}
            )

    else:
        arrow_arrays.append(
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
        arrow_fields.append(
            # Arrow tables must contain at-least one field. To store a
            # non-record layout, we create an un-named field.
            pyarrow.field("", arrow_arrays[-1].type).with_nullable(layout.is_option)
        )

    batch = pyarrow.RecordBatch.from_arrays(
        arrow_arrays, schema=pyarrow.schema(arrow_fields)
    )
    out = pyarrow.Table.from_batches([batch])

    if schema_parameters is None:
        return out
    else:
        return out.replace_schema_metadata(
            {"ak:parameters": json.dumps(schema_parameters)}
        )
