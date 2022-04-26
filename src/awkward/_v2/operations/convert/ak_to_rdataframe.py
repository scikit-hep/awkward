# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def to_rdataframe(arrays, flatlist_as_rvec=True):
    """
    Args:
        arrays (dict): a dictionary of Array-like data (anything #ak.to_layout recognizes).

    Converts an Awkward Array into ROOT Data Frame columns:

    array_x = ak.Array()
    array_y = ak.Array()

    df = ROOT.RDF.MakeAwkwardDataFrame({'x': array_x, 'y': array_y})

    See also #ak.from_rdataframe.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_rdataframe",
        dict(arrays=arrays),
    ):
        return _impl(
            arrays,
            flatlist_as_rvec=flatlist_as_rvec,
        )


def _impl(
    arrays,
    flatlist_as_rvec,
):
    # FIXME: check if there are any arrays or the arrays lengths are equal
    import awkward._v2._connect.rdataframe.to_rdataframe  # noqa: F401

    return ak._v2._connect.rdataframe.to_rdataframe.to_rdataframe(
        arrays,
        flatlist_as_rvec=flatlist_as_rvec,
    )
