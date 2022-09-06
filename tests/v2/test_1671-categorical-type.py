# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array1 = ak._v2.Array(["one", "two", "one", "one"])
    array2 = ak._v2.operations.ak_to_categorical.to_categorical(array1)
    assert array1.type != array2.type
    assert array2.type == ak._v2.types.ArrayType(
        ak._v2.types.ListType(
            ak._v2.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string", "__categorical__": True},
        ),
        4,
    )
