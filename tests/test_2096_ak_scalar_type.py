# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize(
    "array",
    [
        ak.contents.NumpyArray(np.arange(4)),
        ak.contents.IndexedArray(
            ak.index.Index64(np.arange(4, dtype=np.int64)),
            ak.contents.NumpyArray(np.arange(4)),
        ),
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.arange(4, dtype=np.int64)),
            ak.contents.NumpyArray(np.arange(4)),
        ),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.arange(4, dtype=np.int64)),
            ak.contents.NumpyArray(np.arange(3)),
        ),
        ak.contents.ListArray(
            ak.index.Index64(np.arange(3, dtype=np.int64)),
            ak.index.Index64(np.arange(1, 4, dtype=np.int64)),
            ak.contents.NumpyArray(np.arange(3)),
        ),
        ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(12)), size=3),
    ],
)
def test_highlevel_lowlevel(array):
    layout = ak.to_layout(array)
    assert isinstance(ak.type(layout), ak.types.ArrayType)
    # Check that the highlevel of ak.Array yields low level from form
    assert layout.form.type == ak.type(layout).content


def test_array():
    array = ak.Array(["this", {"x": ["is", 1, 2, None]}])
    assert ak.type(array) == array.type
    assert isinstance(array.type, ak.types.ArrayType)


def test_record():
    record = ak.Record({"y": ["this", {"x": ["is", 1, 2, None]}]})
    assert ak.type(record) == record.type
    assert isinstance(record.type, ak.types.ScalarType)


def test_none():
    assert ak.type(None) == ak.types.ScalarType(ak.types.UnknownType())


def test_unknown():
    with pytest.raises(TypeError):
        ak.type(object())


def test_bare_string():
    assert ak.type("hello") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "char"}), 5
    )
    assert ak.type(b"hello") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "byte"}), 5
    )


def test_array_string():
    assert ak.type(["hello"]) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        1,
    )
    assert ak.type([b"hello"]) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "byte"}),
            parameters={"__array__": "bytestring"},
        ),
        1,
    )
