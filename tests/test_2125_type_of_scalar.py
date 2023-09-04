# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test():
    assert ak.type(np.int16()) == ak.types.ScalarType(ak.types.NumpyType("int16"))
    with pytest.raises(TypeError):
        assert ak.type(np.uint32) == ak.types.ScalarType(ak.types.UnknownType())
    assert ak.type(None) == ak.types.ScalarType(ak.types.UnknownType())
    assert ak.type(np.dtype("complex128")) == ak.types.ScalarType(
        ak.types.NumpyType("complex128")
    )
    assert ak.type("hello") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "char"}), 5
    )
    assert ak.type("int16") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "char"}), 5
    )
    assert ak.type(["int16"]) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        1,
    )
