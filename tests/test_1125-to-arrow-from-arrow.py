# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

to_list = ak.operations.to_list


def arrow_round_trip(akarray, paarray, extensionarray, categorical_as_dictionary=False):
    assert to_list(akarray) == paarray.to_pylist()
    akarray2 = ak.from_arrow(paarray, highlevel=False)
    assert to_list(akarray2) == to_list(akarray)
    if extensionarray and categorical_as_dictionary:
        assert akarray2.form.type == akarray.form.type
    akarray3 = ak.from_arrow(
        akarray2.to_arrow(
            extensionarray=extensionarray,
            categorical_as_dictionary=categorical_as_dictionary,
        ),
        highlevel=False,
    )
    if extensionarray and categorical_as_dictionary:
        assert akarray3.form.type == akarray.form.type


def parquet_round_trip(
    akarray, paarray, extensionarray, tmp_path, categorical_as_dictionary=False
):
    filename = os.path.join(tmp_path, "whatever.parquet")
    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    table = pyarrow_parquet.read_table(filename)
    akarray4 = ak.from_arrow(table[0].chunks[0], highlevel=False)
    assert to_list(akarray4) == to_list(akarray)
    if extensionarray and categorical_as_dictionary:
        assert akarray4.form.type == akarray.form.type


@pytest.mark.parametrize("extensionarray", [False, True])
def test_numpyarray(tmp_path, extensionarray):
    akarray = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True, False])),
        ak.contents.NumpyArray(
            np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
        ),
        valid_when=False,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.NumpyArray(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), parameters={"which": "inner"}
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetarray_numpyarray(tmp_path, dtype, list_to32, extensionarray):
    akarray = ak.contents.ListOffsetArray(
        ak.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            parameters={"which": "inner"},
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ListOffsetArray(
        ak.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak.contents.ByteMaskedArray(
            ak.index.Index8(
                np.array(
                    [False, True, False, True, True, False, True, True, False, False],
                    dtype=np.int8,
                )
            ),
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            valid_when=False,
            parameters={"which": "middle"},
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False, True, True, True], dtype=np.int8)),
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            parameters={"which": "middle"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False, True, True, True], dtype=np.int8)),
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
            ak.contents.ByteMaskedArray(
                ak.index.Index8(
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
                ak.contents.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                    parameters={"which": "inner"},
                ),
                valid_when=False,
                parameters={"which": "middle-1"},
            ),
            parameters={"which": "middle-2"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetarray_numpyarray_2(tmp_path, dtype, list_to32, extensionarray):
    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False, True, False, True], dtype=np.int8)),
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            parameters={"which": "middle"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_numpyarray_bool(tmp_path, extensionarray):
    akarray = ak.contents.NumpyArray(
        np.random.randint(0, 2, 14).astype(np.int8).view(np.bool_),
        parameters={"which": "inner"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.random.randint(0, 2, 14).astype(np.int8).view(np.bool_)),
        ak.contents.NumpyArray(
            np.random.randint(0, 2, 14).astype(np.int8).view(np.bool_),
            parameters={"which": "inner"},
        ),
        valid_when=False,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_indexedoptionarray_numpyarray(tmp_path, extensionarray):
    akarray = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([2, 0, 0, -1, 3, 1, 5, -1, 2], dtype=np.int64)),
        ak.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]),
            parameters={"which": "inner"},
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.IndexedArray(
        ak.index.Index64(np.array([2, 0, 0, 3, 1, 5, 2], dtype=np.int64)),
        ak.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]),
            parameters={"which": "inner"},
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_indexedoptionarray_emptyarray(tmp_path, extensionarray):
    akarray = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([-1, -1, -1, -1, -1], dtype=np.int64)),
        ak.contents.EmptyArray(parameters={"which": "inner"}),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("categorical_as_dictionary", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_dictionary_encoding(tmp_path, categorical_as_dictionary, extensionarray):
    akarray = ak.contents.IndexedArray(
        ak.index.Index64(np.array([3, 2, 2, 2, 0, 1, 3], dtype=np.uint64)),
        ak.contents.NumpyArray([0.0, 1.1, 2.2, 3.3], parameters={"which": "inner"}),
        parameters={"__array__": "categorical", "which": "outer"},
    )
    paarray = akarray.to_arrow(
        categorical_as_dictionary=categorical_as_dictionary,
        extensionarray=extensionarray,
    )
    arrow_round_trip(
        akarray,
        paarray,
        extensionarray,
        categorical_as_dictionary=categorical_as_dictionary,
    )

    # https://issues.apache.org/jira/browse/ARROW-14525
    if not (extensionarray and categorical_as_dictionary):
        parquet_round_trip(
            akarray,
            paarray,
            extensionarray,
            tmp_path,
            categorical_as_dictionary=categorical_as_dictionary,
        )


@pytest.mark.parametrize("string_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetraray_string(tmp_path, dtype, string_to32, extensionarray):
    akarray = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "char"}
        ),
        parameters={"__array__": "string", "something": "else"},
    )
    paarray = akarray.to_arrow(string_to32=string_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("bytestring_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_listoffsetraray_bytestring(tmp_path, dtype, bytestring_to32, extensionarray):
    akarray = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "byte"}
        ),
        parameters={"__array__": "bytestring", "something": "else"},
    )
    paarray = akarray.to_arrow(
        bytestring_to32=bytestring_to32, extensionarray=extensionarray
    )
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_regulararray_numpyarray(tmp_path, size, list_to32, extensionarray):
    akarray = ak.contents.RegularArray(
        ak.contents.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            parameters={"which": "inner"},
        ),
        size,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.RegularArray(
        ak.contents.ByteMaskedArray(
            ak.index.Index8(
                np.array(
                    [False, True, False, True, True, False, True, True, False, False],
                    dtype=np.int8,
                )
            ),
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            valid_when=False,
            parameters={"which": "middle"},
        ),
        size,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False], dtype=np.int8)),
        ak.contents.RegularArray(
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
                parameters={"which": "inner"},
            ),
            size,
            parameters={"which": "middle"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)

    # https://issues.apache.org/jira/browse/ARROW-14547
    # parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False], dtype=np.int8)),
        ak.contents.RegularArray(
            ak.contents.ByteMaskedArray(
                ak.index.Index8(
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
                ak.contents.NumpyArray(
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
    paarray = akarray.to_arrow(list_to32=list_to32, extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)

    # https://issues.apache.org/jira/browse/ARROW-14547
    # parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("bytestring_to32", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_regularaarray_bytestring(tmp_path, size, bytestring_to32, extensionarray):
    akarray = ak.contents.RegularArray(
        ak.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "byte"}
        ),
        size,
        parameters={"__array__": "bytestring", "something": "else"},
    )
    paarray = akarray.to_arrow(
        bytestring_to32=bytestring_to32, extensionarray=extensionarray
    )
    arrow_round_trip(akarray, paarray, extensionarray)
    parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_unmaskedarray_numpyarray(tmp_path, extensionarray):
    akarray = ak.contents.UnmaskedArray(
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"})
    )
    paarray = akarray.to_arrow()
    arrow_round_trip(akarray, paarray, True)
    parquet_round_trip(akarray, paarray, True, tmp_path)


@pytest.mark.parametrize("is_tuple", [False, True])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_recordarray(tmp_path, is_tuple, extensionarray):
    akarray = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(
                np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
            ),
            ak.contents.ListOffsetArray(
                ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"which": "inner2"}
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    if not is_tuple or extensionarray:
        arrow_round_trip(akarray, paarray, extensionarray)
        parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.RecordArray(
        [
            ak.contents.ByteMaskedArray(
                ak.index.Index8(np.array([False, True, False]).view(np.int8)),
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                ),
                valid_when=False,
            ),
            ak.contents.ListOffsetArray(
                ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"which": "inner2"}
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    if not is_tuple or extensionarray:
        arrow_round_trip(akarray, paarray, extensionarray)
        parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.RecordArray(
        [
            ak.contents.ByteMaskedArray(
                ak.index.Index8(np.array([False, True, False]).view(np.int8)),
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                ),
                valid_when=False,
            ),
            ak.contents.UnmaskedArray(
                ak.contents.ListOffsetArray(
                    ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                        parameters={"which": "inner2"},
                    ),
                ),
            ),
        ],
        None if is_tuple else ["x", "y"],
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    if not is_tuple or extensionarray:
        arrow_round_trip(akarray, paarray, extensionarray)
        parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([2, 0, -1, 0, 1], dtype=np.int64)),
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(
                    np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                ),
                ak.contents.ListOffsetArray(
                    ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                        parameters={"which": "inner2"},
                    ),
                ),
            ],
            None if is_tuple else ["x", "y"],
            parameters={"which": "outer"},
        ),
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    if not is_tuple or extensionarray:
        arrow_round_trip(akarray, paarray, extensionarray)
        parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([2, 0, -1, 0, 1], dtype=np.int64)),
        ak.contents.RecordArray(
            [
                ak.contents.ByteMaskedArray(
                    ak.index.Index8(np.array([False, True, False]).view(np.int8)),
                    ak.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3]), parameters={"which": "inner1"}
                    ),
                    valid_when=False,
                ),
                ak.contents.ListOffsetArray(
                    ak.index.Index32(np.array([0, 3, 3, 5], dtype=np.int32)),
                    ak.contents.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
                        parameters={"which": "inner2"},
                    ),
                ),
            ],
            None if is_tuple else ["x", "y"],
            parameters={"which": "outer"},
        ),
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    if not is_tuple or extensionarray:
        arrow_round_trip(akarray, paarray, extensionarray)
        parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.skipif(
    not ak._util.numpy_at_least("1.20"), reason="NumPy >= 1.20 required for dates"
)
@pytest.mark.parametrize("extensionarray", [False, True])
def test_numpyarray_datetime(tmp_path, extensionarray):
    # pyarrow doesn't yet support datetime/duration conversions to Parquet.
    # (FIXME: find or create a JIRA ticket.)

    akarray = ak.contents.NumpyArray(
        np.array(
            ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], dtype="datetime64[s]"
        )
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    # parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.NumpyArray(
        np.array(["41", "1", "20"], dtype="timedelta64[s]")
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
    # parquet_round_trip(akarray, paarray, extensionarray, tmp_path)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_unionarray(tmp_path, extensionarray):
    akarray = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 3, 2, 1, 2, 0], dtype=np.int64)),
        [
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2]),
                parameters={"which": "inner1"},
            ),
            ak.contents.NumpyArray(
                np.array([[0], [10], [20], [30]], dtype=np.int32),
                parameters={"which": "inner2"},
            ),
        ],
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)

    # pyarrow doesn't yet support union array conversions to Parquet.
    # (FIXME: find or create a JIRA ticket.)
    # parquet_round_trip(akarray, paarray, extensionarray, tmp_path)

    akarray = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 3, 2, 1, 2, 0], dtype=np.int64)),
        [
            ak.contents.NumpyArray(
                np.array([0.0, 1.1, 2.2]),
                parameters={"which": "inner1"},
            ),
            ak.contents.ByteMaskedArray(
                ak.index.Index8(np.array([False, False, True, False]).view(np.int8)),
                ak.contents.NumpyArray(
                    np.array([[0], [10], [20], [30]], dtype=np.int32),
                    parameters={"which": "inner2"},
                ),
                valid_when=False,
                parameters={"which": "middle"},
            ),
        ],
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)

    akarray = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(
            np.array([False, True, False, True, False, True, True]).view(np.int8)
        ),
        ak.contents.UnionArray(
            ak.index.Index8(np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.int8)),
            ak.index.Index64(np.array([0, 1, 3, 2, 1, 2, 0], dtype=np.int64)),
            [
                ak.contents.NumpyArray(
                    np.array([0.0, 1.1, 2.2]), parameters={"which": "inner1"}
                ),
                ak.contents.NumpyArray(
                    np.array([[0], [10], [20], [30]], dtype=np.int32),
                    parameters={"which": "inner2"},
                ),
            ],
            parameters={"which": "middle"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)

    akarray = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(
            np.array([False, True, False, True, False, True, True]).view(np.int8)
        ),
        ak.contents.UnionArray(
            ak.index.Index8(np.array([0, 0, 1, 1, 1, 0, 1], dtype=np.int8)),
            ak.index.Index64(np.array([0, 1, 3, 2, 1, 2, 0], dtype=np.int64)),
            [
                ak.contents.NumpyArray(
                    np.array([0.0, 1.1, 2.2]), parameters={"which": "inner1"}
                ),
                ak.contents.ByteMaskedArray(
                    ak.index.Index8(
                        np.array([False, False, True, False]).view(np.int8)
                    ),
                    ak.contents.NumpyArray(
                        np.array([[0], [10], [20], [30]], dtype=np.int32),
                        parameters={"which": "inner2"},
                    ),
                    valid_when=False,
                    parameters={"which": "middle1"},
                ),
            ],
            parameters={"which": "middle2"},
        ),
        valid_when=True,
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(extensionarray=extensionarray)
    arrow_round_trip(akarray, paarray, extensionarray)
