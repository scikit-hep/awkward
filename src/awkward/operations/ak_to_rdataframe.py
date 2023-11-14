# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._dispatch import high_level_function

__all__ = ("to_rdataframe",)

cpu = NumpyBackend.instance()


@high_level_function()
def to_rdataframe(arrays, *, flatlist_as_rvec=True):
    """
    Args:
        arrays (dict of arrays): Each value in this dict can be any array-like data
            that #ak.to_layout recognizes, but they must all have the same length.
        flatlist_as_rvec (bool): If True, lists of primitive types (numbers, booleans, etc.)
            are presented to C++ as `ROOT::RVec<primitive>`, but all other types use
            Awkward Array's custom C++ classes. If False, even these "flat" lists use
            Awkward Array's custom C++ classes.

    Converts an Awkward Array into ROOT Data Frame columns:

        >>> x = ak.Array([
        ...     [1.1, 2.2, 3.3],
        ...     [],
        ...     [4.4, 5.5],
        ... ])
        >>> y = ak.Array([
        ...     {"a": 1.1, "b": [1]},
        ...     {"a": 2.2, "b": [2, 1]},
        ...     {"a": 3.3, "b": [3, 2, 1]},
        ... ])

        >>> rdf = ak.to_rdataframe({"x": x, "y": y})
        >>> rdf.Define("z", "ROOT::VecOps::Sum(x) + y.a() + y.b()[0]").AsNumpy(["z"])
        {'z': ndarray([ 8.7,  4.2, 16.2])}

        >>> ak.sum(x, axis=-1) + y.a + y.b[:, 0]
        <Array [8.7, 4.2, 16.2] type='3 * float64'>

    See also #ak.from_rdataframe.
    """
    # Dispatch
    yield arrays.values()

    # Implementation
    return _impl(
        arrays,
        flatlist_as_rvec=flatlist_as_rvec,
    )


def _impl(
    arrays,
    flatlist_as_rvec,
):
    import awkward._connect.rdataframe.to_rdataframe  # noqa: F401

    if not isinstance(arrays, Mapping):
        raise TypeError("'arrays' must be a dict (to provide C++ names for the arrays)")
    elif not all(isinstance(name, str) for name in arrays):
        raise TypeError(
            "keys of 'arrays' dict must be strings (to provide C++ names for the arrays)"
        )
    elif len(arrays) == 0:
        raise TypeError("'arrays' must contain at least one array")

    layouts = {}
    length = None
    for name, array in arrays.items():
        layouts[name] = ak.operations.ak_to_layout._impl(
            array,
            allow_record=False,
            allow_unknown=False,
            none_policy="error",
            regulararray=True,
            use_from_iter=True,
            primitive_policy="forbid",
            string_policy="as-characters",
        ).to_backend(cpu)
        if length is None:
            length = layouts[name].length
        elif length != layouts[name].length:
            raise ValueError("lengths of 'arrays' must all be the same")

    return ak._connect.rdataframe.to_rdataframe.to_rdataframe(
        layouts,
        length,
        flatlist_as_rvec=flatlist_as_rvec,
    )
