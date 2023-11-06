# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_to_categorical():
    pytest.importorskip("pyarrow")

    array1 = ak.Array(["one", "two", "one", "one"])
    array2 = ak.str.to_categorical(array1)
    assert array1.type != array2.type
    assert array2.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string", "__categorical__": True},
        ),
        4,
    )


def test_categorical_type():
    pytest.importorskip("pyarrow")

    array1 = ak.Array(["one", "two", "one", "one"])
    array2 = ak.str.to_categorical(array1)
    assert array1.type != array2.type
    assert array2.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string", "__categorical__": True},
        ),
        4,
    )
