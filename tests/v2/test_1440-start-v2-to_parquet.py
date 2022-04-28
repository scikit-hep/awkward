# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

to_list = ak._v2.operations.convert.to_list


def parquet_round_trip(akarray, extensionarray, tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    ak._v2.to_parquet(akarray, filename, extensionarray=extensionarray)
    akarray2 = ak._v2.from_parquet(filename)

    assert to_list(akarray2) == to_list(akarray)
    if extensionarray:
        print("read back")
        akarray2.type.show()
        print("original")
        akarray.type.show()

        assert akarray2.type == akarray.type


@pytest.mark.parametrize("extensionarray", [False, True])
def test_numpyarray(tmp_path, extensionarray):
    akarray = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.array([False, True, False])),
        ak._v2.contents.NumpyArray(
            np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
        ),
        valid_when=False,
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.NumpyArray(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), parameters={"which": "inner"}
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetarray_numpyarray(tmp_path, dtype, list_to32, extensionarray):
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak._v2.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            parameters={"which": "inner"},
        ),
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    # akarray = ak._v2.contents.ListOffsetArray(
    #     ak._v2.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
    #     ak._v2.contents.ByteMaskedArray(
    #         ak._v2.index.Index8(
    #             np.array(
    #                 [False, True, False, True, True, False, True, True, False, False],
    #                 dtype=np.int8,
    #             )
    #         ),
    #         ak._v2.contents.NumpyArray(
    #             np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
    #             parameters={"which": "inner"},
    #         ),
    #         valid_when=False,
    #         parameters={"which": "middle"},
    #     ),
    #     parameters={"which": "outer"},
    # )
    # parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    # akarray = ak._v2.contents.ByteMaskedArray(
    #     ak._v2.index.Index8(np.array([True, False, True, True, True], dtype=np.int8)),
    #     ak._v2.contents.ListOffsetArray(
    #         ak._v2.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
    #         ak._v2.contents.NumpyArray(
    #             np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
    #             parameters={"which": "inner"},
    #         ),
    #         parameters={"which": "middle"},
    #     ),
    #     valid_when=True,
    #     parameters={"which": "outer"},
    # )
    # parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    # akarray = ak._v2.contents.ByteMaskedArray(
    #     ak._v2.index.Index8(np.array([True, False, True, True, True], dtype=np.int8)),
    #     ak._v2.contents.ListOffsetArray(
    #         ak._v2.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
    #         ak._v2.contents.ByteMaskedArray(
    #             ak._v2.index.Index8(
    #                 np.array(
    #                     [
    #                         False,
    #                         True,
    #                         False,
    #                         True,
    #                         True,
    #                         False,
    #                         True,
    #                         True,
    #                         False,
    #                         False,
    #                     ],
    #                     dtype=np.int8,
    #                 )
    #             ),
    #             ak._v2.contents.NumpyArray(
    #                 np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
    #                 parameters={"which": "inner"},
    #             ),
    #             valid_when=False,
    #             parameters={"which": "middle-1"},
    #         ),
    #         parameters={"which": "middle-2"},
    #     ),
    #     valid_when=True,
    #     parameters={"which": "outer"},
    # )
    # parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetarray_numpyarray_2(tmp_path, dtype, list_to32, extensionarray):
    akarray = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.array([True, False, True, False, True], dtype=np.int8)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
            ak._v2.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            parameters={"which": "middle"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_numpyarray_bool(tmp_path, extensionarray):
    akarray = ak._v2.contents.NumpyArray(
        np.random.randint(0, 2, 14).astype(np.int8).view(np.bool_),
        parameters={"which": "inner"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.random.randint(0, 2, 14).astype(np.int8).view(np.bool_)),
        ak._v2.contents.NumpyArray(
            np.random.randint(0, 2, 14).astype(np.int8).view(np.bool_),
            parameters={"which": "inner"},
        ),
        valid_when=False,
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_indexedoptionarray_numpyarray(tmp_path, extensionarray):
    akarray = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.array([2, 0, 0, -1, 3, 1, 5, -1, 2], dtype=np.int64)),
        ak._v2.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]),
            parameters={"which": "inner"},
        ),
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.IndexedArray(
        ak._v2.index.Index64(np.array([2, 0, 0, 3, 1, 5, 2], dtype=np.int64)),
        ak._v2.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]),
            parameters={"which": "inner"},
        ),
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_indexedoptionarray_emptyarray(tmp_path, extensionarray):
    akarray = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.array([-1, -1, -1, -1, -1], dtype=np.int64)),
        ak._v2.contents.EmptyArray(parameters={"which": "inner"}),
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("categorical_as_dictionary", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_dictionary_encoding(tmp_path, categorical_as_dictionary, extensionarray):
    akarray = ak._v2.contents.IndexedArray(
        ak._v2.index.Index64(np.array([3, 2, 2, 2, 0, 1, 3], dtype=np.uint64)),
        ak._v2.contents.NumpyArray([0.0, 1.1, 2.2, 3.3], parameters={"which": "inner"}),
        parameters={"__array__": "categorical", "which": "outer"},
    )

    # https://issues.apache.org/jira/browse/ARROW-14525
    if not (extensionarray and categorical_as_dictionary):
        parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("string_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetraray_string(tmp_path, dtype, string_to32, extensionarray):
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak._v2.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "char"}
        ),
        parameters={"__array__": "string", "something": "else"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("bytestring_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetraray_bytestring(tmp_path, dtype, bytestring_to32, extensionarray):
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak._v2.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "byte"}
        ),
        parameters={"__array__": "bytestring", "something": "else"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_regulararray_numpyarray(tmp_path, size, list_to32, extensionarray):
    akarray = ak._v2.contents.RegularArray(
        ak._v2.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            parameters={"which": "inner"},
        ),
        size,
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.RegularArray(
        ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index8(
                np.array(
                    [False, True, False, True, True, False, True, True, False, False],
                    dtype=np.int8,
                )
            ),
            ak._v2.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            valid_when=False,
            parameters={"which": "middle"},
        ),
        size,
        parameters={"which": "outer"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.array([True, False], dtype=np.int8)),
        ak._v2.contents.RegularArray(
            ak._v2.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            size,
            parameters={"which": "middle"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )

    # https://issues.apache.org/jira/browse/ARROW-14547
    # parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.array([True, False], dtype=np.int8)),
        ak._v2.contents.RegularArray(
            ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(
                    np.array(
                        [
                            False,
                            True,
                            False,
                            True,
                            True,
                            False,
                            True,
                            True,
                            False,
                            False,
                        ],
                        dtype=np.int8,
                    )
                ),
                ak._v2.contents.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                    parameters={"which": "inner"},
                ),
                valid_when=False,
                parameters={"which": "middle-1"},
            ),
            size,
            parameters={"which": "middle-2"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )

    # https://issues.apache.org/jira/browse/ARROW-14547
    # parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("bytestring_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_regularaarray_bytestring(tmp_path, size, bytestring_to32, extensionarray):
    akarray = ak._v2.contents.RegularArray(
        ak._v2.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "byte"}
        ),
        size,
        parameters={"__array__": "bytestring", "something": "else"},
    )
    parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_unmaskedarray_numpyarray(tmp_path, extensionarray):
    akarray = ak._v2.contents.UnmaskedArray(
        ak._v2.contents.NumpyArray(
            np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
        )
    )
    parquet_round_trip(ak._v2.Array(akarray), True, tmp_path)


@pytest.mark.parametrize("is_tuple", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_recordarray(tmp_path, is_tuple, extensionarray):
    akarray = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.NumpyArray(
                np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
            ),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                ak._v2.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"which": "inner2"}
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"which": "outer"},
    )
    if not is_tuple or extensionarray:
        parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(np.array([False, True, False]).view(np.int8)),
                ak._v2.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                ),
                valid_when=False,
            ),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                ak._v2.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"which": "inner2"}
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"which": "outer"},
    )
    if not is_tuple or extensionarray:
        parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(np.array([False, True, False]).view(np.int8)),
                ak._v2.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                ),
                valid_when=False,
            ),
            ak._v2.contents.UnmaskedArray(
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak._v2.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                        parameters={"which": "inner2"},
                    ),
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"which": "outer"},
    )
    if not is_tuple or extensionarray:
        parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.array([2, 0, -1, 0, 1], dtype=np.int64)),
        ak._v2.contents.RecordArray(
            [
                ak._v2.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                ),
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak._v2.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                        parameters={"which": "inner2"},
                    ),
                ),
            ],
            None if is_tuple else ["x", "y"],
            parameters={"which": "outer"},
        ),
    )
    if not is_tuple or extensionarray:
        parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)

    akarray = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.array([2, 0, -1, 0, 1], dtype=np.int64)),
        ak._v2.contents.RecordArray(
            [
                ak._v2.contents.ByteMaskedArray(
                    ak._v2.index.Index8(np.array([False, True, False]).view(np.int8)),
                    ak._v2.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                    ),
                    valid_when=False,
                ),
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak._v2.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                        parameters={"which": "inner2"},
                    ),
                ),
            ],
            None if is_tuple else ["x", "y"],
            parameters={"which": "outer"},
        ),
    )
    if not is_tuple or extensionarray:
        parquet_round_trip(ak._v2.Array(akarray), extensionarray, tmp_path)
