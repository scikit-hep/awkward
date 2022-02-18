# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_arrow(array, conservative_optiontype=False, highlevel=True, behavior=None):
    """
    Args:
        array (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`,
            or `pyarrow.Table`): Apache Arrow array to convert into an
            Awkward Array.
        conservative_optiontype (bool): If enabled and the optionalness of a type
            can't be determined (i.e. not within an `pyarrow.field` and the Arrow array
            has no Awkward metadata), assume that it is option-type with a blank
            BitMaskedArray.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
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
    import awkward._v2._connect.pyarrow

    out = awkward._v2._connect.pyarrow.handle_arrow(
        array, conservative_optiontype=conservative_optiontype, pass_empty_field=True
    )
    return ak._v2._util.wrap(out, behavior, highlevel)
