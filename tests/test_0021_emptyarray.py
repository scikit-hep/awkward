# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_unknown():
    i = ak.index.Index64(np.array([0, 0, 0, 0], dtype=np.int64))
    e = ak.contents.EmptyArray()
    a = ak.contents.ListOffsetArray(i, e)
    assert to_list(a) == [[], [], []]
    assert str(ak.operations.type(a)) == "var * unknown"
    assert ak.operations.type(a) == ak.types.ListType(ak.types.UnknownType())
    assert not ak.operations.type(a) == ak.types.NumpyType("float64")

    i = ak.index.Index64(np.array([0, 0, 0, 0, 0, 0], dtype=np.int64))
    ii = ak.index.Index64(np.array([0, 0, 2, 5], dtype=np.int64))
    a = ak.contents.ListOffsetArray(i, e)
    a = ak.contents.ListOffsetArray(ii, a)

    assert to_list(a) == [[], [[], []], [[], [], []]]
    assert str(ak.operations.type(a)) == "var * var * unknown"
    assert ak.operations.type(a) == ak.types.ListType(
        ak.types.ListType(ak.types.UnknownType())
    )


def test_unknown_arraybuilder():
    a = ak.highlevel.ArrayBuilder()
    a.begin_list()
    a.end_list()
    a.begin_list()
    a.end_list()
    a.begin_list()
    a.end_list()
    assert to_list(a) == [[], [], []]
    assert str(ak.operations.type(a)) == "3 * var * unknown"
    assert ak.operations.type(a) == ak.types.ArrayType(
        ak.types.ListType(ak.types.UnknownType()), 3
    )
    assert not ak.operations.type(a) == ak.types.NumpyType("float64")

    a = a.snapshot()
    assert to_list(a) == [[], [], []]
    assert str(ak.operations.type(a)) == "3 * var * unknown"
    assert ak.operations.type(a) == ak.types.ArrayType(
        ak.types.ListType(ak.types.UnknownType()), 3
    )
    assert not ak.operations.type(a) == ak.types.ArrayType(
        ak.types.NumpyType("float64"), 3
    )


def test_getitem():
    a = ak.operations.from_iter([[], [[], []], [[], [], []]], highlevel=False)

    assert to_list(a[2]) == [[], [], []]
    assert a.to_typetracer()[2].form == a[2].form

    assert to_list(a[2, 1]) == []
    assert a.to_typetracer()[2, 1].form == a[2, 1].form
    with pytest.raises(IndexError):
        a[2, 1, 0]
    assert to_list(a[2, 1][()]) == []
    assert a.to_typetracer()[2, 1][()].form == a[2, 1][()].form
    with pytest.raises(IndexError):
        a[2, 1][0]
    assert to_list(a[2, 1][100:200]) == []
    assert a.to_typetracer()[2, 1][100:200].form == a[2, 1][100:200].form
    assert to_list(a[2, 1, 100:200]) == []
    assert a.to_typetracer()[2, 1, 100:200].form == a[2, 1, 100:200].form
    assert to_list(a[2, 1][np.array([], dtype=np.int64)]) == []
    assert (
        a.to_typetracer()[2, 1][np.array([], dtype=np.int64)].form
        == a[2, 1][np.array([], dtype=np.int64)].form
    )
    assert to_list(a[2, 1, np.array([], dtype=np.int64)]) == []
    assert (
        a.to_typetracer()[2, 1, np.array([], dtype=np.int64)].form
        == a[2, 1, np.array([], dtype=np.int64)].form
    )
    with pytest.raises(IndexError):
        a[2, 1, np.array([0], dtype=np.int64)]
    with pytest.raises(IndexError):
        a[2, 1][100:200, 0]
    with pytest.raises(IndexError):
        a[2, 1][100:200, 200:300]

    assert to_list(a[2, 1][100:200, np.array([], dtype=np.int64)]) == []
    assert (
        a.to_typetracer()[2, 1][100:200, np.array([], dtype=np.int64)].form
        == a[2, 1][100:200, np.array([], dtype=np.int64)].form
    )

    assert to_list(a[1:, 1:]) == [[[]], [[], []]]
    assert a.to_typetracer()[1:, 1:].form == a[1:, 1:].form
    with pytest.raises(IndexError):
        a[1:, 1:, 0]


def test_unknown2():
    a = ak.operations.from_json("[[], [], []]", highlevel=False)
    assert a.to_list() == [[], [], []]
    assert str(a.form.type) == "var * unknown"
    assert a.form.type == ak.types.ListType(ak.types.UnknownType())
    assert not a.form.type == ak.types.NumpyType("float64")

    a = ak.operations.from_json("[[], [[], []], [[], [], []]]", highlevel=False)
    assert a.to_list() == [[], [[], []], [[], [], []]]
    assert str(a.form.type) == "var * var * unknown"
    assert a.form.type == ak.types.ListType(ak.types.ListType(ak.types.UnknownType()))

    a = ak.highlevel.ArrayBuilder()
    a.begin_list()
    a.end_list()
    a.begin_list()
    a.end_list()
    a.begin_list()
    a.end_list()
    assert a.to_list() == [[], [], []]
    assert str(a.type) == "3 * var * unknown"
    assert a.type == ak.types.ArrayType(ak.types.ListType(ak.types.UnknownType()), 3)
    assert not a.type == ak.types.NumpyType("float64")

    a = a.snapshot()
    assert a.to_list() == [[], [], []]
    assert str(a.type) == "3 * var * unknown"
    assert a.type == ak.types.ArrayType(ak.types.ListType(ak.types.UnknownType()), 3)
    assert not a.type == ak.types.NumpyType("float64")


def test_from_json_getitem():
    a = ak.operations.from_json("[]")
    a = ak.operations.from_json("[[], [[], []], [[], [], []]]")
    assert a[2].to_list() == [[], [], []]

    assert a[2, 1].to_list() == []
    with pytest.raises(IndexError) as excinfo:
        a[2, 1, 0]
    assert "index out of range while attempting to get index 0" in str(excinfo.value)
    assert a[2, 1][()].to_list() == []
    with pytest.raises(IndexError) as excinfo:
        a[2, 1][0]
    assert (
        "<Array [] type='0 * unknown'>\n\nwith\n\n    0\n\nat inner EmptyArray of length 0, using sub-slice 0.\n\nError details: array is empty."
        in str(excinfo.value)
    )
    assert a[2, 1][100:200].to_list() == []
    assert a[2, 1, 100:200].to_list() == []
    assert a[2, 1][np.array([], dtype=np.int64)].to_list() == []
    assert a[2, 1, np.array([], dtype=np.int64)].to_list() == []
    with pytest.raises(IndexError) as excinfo:
        a[2, 1, np.array([0], dtype=np.int64)]
    assert "index out of range while attempting to get index 0" in str(excinfo.value)
    with pytest.raises(IndexError) as excinfo:
        a[2, 1][100:200, 0]
    assert (
        "<Array [] type='0 * unknown'>\n\nwith\n\n    (100:200, 0)\n\nat inner EmptyArray of length 0, using sub-slice 0.\n\nError details: array is empty."
        in str(excinfo.value)
    )
    with pytest.raises(IndexError) as excinfo:
        a[2, 1][100:200, 200:300]
    assert (
        "<Array [] type='0 * unknown'>\n\nwith\n\n    (100:200, 200:300)\n\nat inner EmptyArray of length 0, using sub-slice 200:300.\n\nError details: array is empty."
        in str(excinfo.value)
    )

    # FIXME: Failed: DID NOT RAISE <class 'IndexError'>
    # with pytest.raises(IndexError) as excinfo:
    #     a[2, 1][100:200, np.array([], dtype=np.int64)]
    # assert ", too many dimensions in slice" in str(excinfo.value)

    assert a[1:, 1:].to_list() == [[[]], [[], []]]
    with pytest.raises(IndexError) as excinfo:
        a[1:, 1:, 0]
    assert "index out of range while attempting to get index 0" in str(excinfo.value)
