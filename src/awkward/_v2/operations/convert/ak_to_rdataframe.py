# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def to_rdataframe(
    array,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array : Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an Awkward Array into ROOT Data Frame columns:

    array_x = ak.Array()
    array_y = ak.Array()

    df = ROOT.RDF.MakeAwkwardDataFrame({'x': array_x, 'y': array_y})

    See also #ak.from_rdataframe.
    """
    import awkward._v2._connect.rdataframe.to_rdataframe  # noqa: F401

    raise ak._v2._util.error(NotImplementedError)
