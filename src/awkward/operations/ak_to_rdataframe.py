# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Mapping

import awkward as ak


def to_rdataframe(arrays, flatlist_as_rvec=True):
    """
    Args:
        arrays (dict of str \u2192 arrays): A dictionary of Array-like data (anything
            #ak.to_layout recognizes). All of these arrays must have the same length.
        flatlist_as_rvec (bool): If True, lists of primitive types (numbers, booleans, etc.)
            are presented to C++ as `ROOT::RVec<primitive>`, but all other types use
            Awkward Array's custom C++ classes. If False, even these "flat" lists use
            Awkward Array's custom C++ classes.

    Converts an Awkward Array into ROOT Data Frame columns:

        >>> x = ak._v2.Array([
        ...     [1.1, 2.2, 3.3],
        ...     [],
        ...     [4.4, 5.5],
        ... ])
        >>> y = ak._v2.Array([
        ...     {"a": 1.1, "b": [1]},
        ...     {"a": 2.2, "b": [2, 1]},
        ...     {"a": 3.3, "b": [3, 2, 1]},
        ... ])

        >>> rdf = ak._v2.to_rdataframe({"x": x, "y": y})
        >>> rdf.Define("z", "ROOT::VecOps::Sum(x) + y.a() + y.b()[0]").AsNumpy(["z"])
        {'z': ndarray([ 8.7,  4.2, 16.2])}

        >>> ak._v2.sum(x, axis=-1) + y.a + y.b[:, 0]
        <Array [8.7, 4.2, 16.2] type='3 * float64'>

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
    import awkward._v2._connect.rdataframe.to_rdataframe  # noqa: F401

    if not isinstance(arrays, Mapping):
        raise ak._v2._util.error(
            TypeError("'arrays' must be a dict (to provide C++ names for the arrays)")
        )
    elif not all(ak._v2._util.isstr(name) for name in arrays):
        raise ak._v2._util.error(
            TypeError(
                "keys of 'arrays' dict must be strings (to provide C++ names for the arrays)"
            )
        )
    elif len(arrays) == 0:
        raise ak._v2._util.error(TypeError("'arrays' must contain at least one array"))

    layouts = {}
    length = None
    for name, array in arrays.items():
        layouts[name] = ak._v2.operations.ak_to_layout.to_layout(
            array, allow_record=False, allow_other=False
        )
        if length is None:
            length = layouts[name].length
        elif length != layouts[name].length:
            raise ak._v2._util.error(
                ValueError("lengths of 'arrays' must all be the same")
            )

    return ak._v2._connect.rdataframe.to_rdataframe.to_rdataframe(
        layouts,
        length,
        flatlist_as_rvec=flatlist_as_rvec,
    )
