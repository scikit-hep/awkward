# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_rdataframe(data_frame, column, column_as_record=True):
    """
    Args:
        data_frame (`ROOT.RDataFrame`): ROOT RDataFrame to convert into an
             Awkward Array.
         column (str): A column to be converted to Awkward Array.
         column_as_records (bool): If True, the column converted as records.

     Converts ROOT Data Frame columns into an Awkward Array.

     See also #ak.to_rdataframe.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_rdataframe",
        dict(column=column),
    ):
        return _impl(
            data_frame,
            column,
            column_as_record,
        )


def _impl(
    data_frame,
    column,
    column_as_record,
):
    import awkward._v2._connect.rdataframe.from_rdataframe  # noqa: F401

    return ak._v2._connect.rdataframe.from_rdataframe.from_rdataframe(
        data_frame,
        column,
        column_as_record,
    )
