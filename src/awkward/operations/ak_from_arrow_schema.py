# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def from_arrow_schema(schema):
    """
    Args:
        schema (`pyarrow.Schema`): Apache Arrow schema to convert into an Awkward Form.

    Converts an Apache Arrow schema into an Awkward Form.

    See also #ak.to_arrow, #ak.to_arrow_table, #ak.from_arrow, #ak.to_parquet, #ak.from_parquet.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_arrow_schema",
        dict(schema=schema),
    ):
        return _impl(schema)


def _impl(schema):
    import awkward._connect.pyarrow

    return awkward._connect.pyarrow.form_handle_arrow(schema, pass_empty_field=True)
