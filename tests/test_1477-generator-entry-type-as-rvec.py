# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_NumpyArray_shape():
    ak_array_in = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert (
        data_frame.GetColumnType("x")
        == "ROOT::VecOps::RVec<ROOT::VecOps::RVec<int64_t>>"
    )


def test_RegularArray_NumpyArray():
    ak_array_in = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"


def test_ListArray_NumpyArray():
    ak_array_in = ak._v2.contents.listarray.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"


def test_ListOffsetArray_NumpyArray():
    ak_array_in = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"


def test_RecordArray_NumpyArray():
    ak_array_in_one = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )
    data_frame_one = ak._v2.to_rdataframe({"one": ak_array_in_one})
    assert str(data_frame_one.GetColumnType("one")).startswith(
        "awkward::Record_Something_"
    )


def test_RecordArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        None,
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert str(data_frame_two.GetColumnType("two")).startswith("awkward::Record_")


def test_RecordArray_NumpyArray_three():
    ak_array_three = ak._v2.contents.recordarray.RecordArray([], [], 10)
    data_frame_three = ak._v2.to_rdataframe({"three": ak_array_three})
    assert str(data_frame_three.GetColumnType("three")).startswith("awkward::Record_")


def test_RecordArray_NumpyArray_four():
    ak_array_four = ak._v2.contents.recordarray.RecordArray([], None, 10)
    data_frame_four = ak._v2.to_rdataframe({"four": ak_array_four})
    assert str(data_frame_four.GetColumnType("four")).startswith("awkward::Record_")


def test_IndexedArray_NumpyArray():
    ak_array_in = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "double"


def test_IndexedOptionArray_NumpyArray():
    ak_array_in = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "std::optional<double>"


def test_ByteMaskedArray_NumpyArray():
    ak_array_one = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    data_frame_one = ak._v2.to_rdataframe({"one": ak_array_one})
    assert data_frame_one.GetColumnType("one") == "std::optional<double>"


def test_ByteMaskedArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert data_frame_two.GetColumnType("two") == "std::optional<double>"


def test_BitMaskedArray_NumpyArray():
    ak_array_one = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    data_frame_one = ak._v2.to_rdataframe({"one": ak_array_one})
    assert data_frame_one.GetColumnType("one") == "std::optional<double>"


def test_BitMaskedArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert data_frame_two.GetColumnType("two") == "std::optional<double>"


def test_BitMaskedArray_NumpyArray_three():
    ak_array_three = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    data_frame_three = ak._v2.to_rdataframe({"three": ak_array_three})
    assert data_frame_three.GetColumnType("three") == "std::optional<double>"


def test_BitMaskedArray_NumpyArray_four():
    ak_array_four = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    data_frame_four = ak._v2.to_rdataframe({"four": ak_array_four})
    assert data_frame_four.GetColumnType("four") == "std::optional<double>"


def test_UnmaskedArray_NumpyArray():
    ak_array_in = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "std::optional<double>"


def test_nested_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([999.0, 0.0, 1.1, 2.2, 3.3]),
            parameters={"some": "stuff", "other": [1, 2, "three"]},
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"


def test_nested_NumpyArray_shape():
    data = np.full((3, 3, 5), 999, dtype=np.int64)
    data[1:3] = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)

    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(data),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::RegularArray_")


def test_nested_RegularArray_NumpyArray():
    ak_array_one = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak._v2.contents.regulararray.RegularArray(
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([999, 999, 999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
            3,
        ),
    )
    data_frame_one = ak._v2.to_rdataframe({"one": ak_array_one})
    assert str(data_frame_one.GetColumnType("one")).startswith("awkward::RegularArray_")


def test_nested_RegularArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak._v2.contents.regulararray.RegularArray(
            ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(np.dtype(np.float64)),
            0,
            zeros_length=11,
        ),
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert str(data_frame_two.GetColumnType("two")).startswith("awkward::RegularArray_")


def test_nested_ListArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 4], dtype=np.int64)),
        ak._v2.contents.listarray.ListArray(
            ak._v2.index.Index(np.array([999, 4, 100, 1], np.int64)),
            ak._v2.index.Index(np.array([999, 7, 100, 3, 200], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
            ),
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::ListArray_")


def test_nested_ListOffsetArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.listoffsetarray.ListOffsetArray(
            ak._v2.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::ListArray_")


def test_nested_RecordArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([999, 0, 1, 2, 3, 4], np.int64)
                ),
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
                ),
            ],
            ["x", "y"],
            parameters={"__record__": "Something"},
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith(
        "awkward::RecordArray_Something_"
    )


def test_nested_RecordArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([999, 0, 1, 2, 3, 4], np.int64)
                ),
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
                ),
            ],
            None,
        ),
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert str(data_frame_two.GetColumnType("two")).startswith("awkward::RecordArray_")


def test_nested_RecordArray_NumpyArray_three():
    ak_array_three = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray([], [], 11),
    )
    data_frame_three = ak._v2.to_rdataframe({"three": ak_array_three})
    assert str(data_frame_three.GetColumnType("three")).startswith(
        "awkward::RecordArray_"
    )


def test_nested_RecordArray_NumpyArray_four():
    ak_array_four = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak._v2.contents.recordarray.RecordArray([], None, 11),
    )
    data_frame_four = ak._v2.to_rdataframe({"four": ak_array_four})
    assert str(data_frame_four.GetColumnType("four")).startswith(
        "awkward::RecordArray_"
    )


def test_nested_IndexedArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak._v2.contents.indexedarray.IndexedArray(
            ak._v2.index.Index(np.array([999, 2, 2, 0, 1, 4, 5, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::IndexedArray_")


def test_nested_IndexedOptionArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            ak._v2.index.Index(np.array([999, 2, 2, -1, 1, -1, 5, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::IndexedOptionArray_")


def test_nested_ByteMaskedArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak._v2.contents.bytemaskedarray.ByteMaskedArray(
            ak._v2.index.Index(np.array([123, 1, 0, 1, 0, 1], np.int8)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([999, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            ),
            valid_when=True,
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::ByteMaskedArray_")


def test_nested_ByteMaskedArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak._v2.contents.bytemaskedarray.ByteMaskedArray(
            ak._v2.index.Index(np.array([123, 0, 1, 0, 1, 0], np.int8)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([999, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            ),
            valid_when=False,
        ),
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert str(data_frame_two.GetColumnType("two")).startswith(
        "awkward::ByteMaskedArray_"
    )


def test_nested_BitMaskedArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak._v2.contents.bitmaskedarray.BitMaskedArray(
            ak._v2.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=True,
            length=14,
            lsb_order=False,
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::BitMaskedArray_")


def test_nested_BitMaskedArray_NumpyArray_two():
    ak_array_two = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak._v2.contents.bitmaskedarray.BitMaskedArray(
            ak._v2.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=False,
            length=14,
            lsb_order=False,
        ),
    )
    data_frame_two = ak._v2.to_rdataframe({"two": ak_array_two})
    assert str(data_frame_two.GetColumnType("two")).startswith(
        "awkward::BitMaskedArray_"
    )


def test_nested_BitMaskedArray_NumpyArray_three():
    ak_array_three = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak._v2.contents.bitmaskedarray.BitMaskedArray(
            ak._v2.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            0,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=True,
            length=14,
            lsb_order=True,
        ),
    )
    data_frame_three = ak._v2.to_rdataframe({"three": ak_array_three})
    assert str(data_frame_three.GetColumnType("three")).startswith(
        "awkward::BitMaskedArray_"
    )


def test_nested_BitMaskedArray_NumpyArray_four():
    ak_array_four = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak._v2.contents.bitmaskedarray.BitMaskedArray(
            ak._v2.index.Index(
                np.packbits(
                    np.array(
                        [
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            1,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=False,
            length=14,
            lsb_order=True,
        ),
    )
    data_frame_four = ak._v2.to_rdataframe({"four": ak_array_four})
    assert str(data_frame_four.GetColumnType("four")).startswith(
        "awkward::BitMaskedArray_"
    )


def test_nested_UnmaskedArray_NumpyArray():
    ak_array_in = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.unmaskedarray.UnmaskedArray(
            ak._v2.contents.numpyarray.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )
    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert str(data_frame.GetColumnType("x")).startswith("awkward::UnmaskedArray_")
