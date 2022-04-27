# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def from_rdataframe(data_frame, columns=None, exclude=None, columns_as_records=True):
    """
    Args:
        data_frame (`ROOT.RDataFrame`): ROOT RDataFrame to convert into an
             Awkward Array.
         columns (None or str): List of columns to be converted to Awkward Arrays.
         exclude (None or str): List of columns to be excluded from the conversion.
         columns_as_records (bool): If True, columns converted as records.

     Converts ROOT Data Frame columns into an Awkward Array.

     See also #ak.to_rdataframe.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_rdataframe",
        dict(columns=columns),
    ):
        return _impl(
            data_frame,
            columns,
            exclude,
            columns_as_records,
        )


def _impl(
    data_frame,
    columns,
    exclude,
    columns_as_records,
):
    import awkward._v2._connect.rdataframe.from_rdataframe  # noqa: F401

    return ak._v2._connect.rdataframe.from_rdataframe.from_rdataframe(
        data_frame,
        columns,
        exclude,
        columns_as_records,
    )
