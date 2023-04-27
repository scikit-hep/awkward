# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy
import pytest

import awkward as ak


def test_record():
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{x: var * int64}", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                )
            ],
            ["x"],
        ),
    )

    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 0]}], regulararray=False),
        ak.types.from_datashape("{x: var * bool}", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 0], dtype=numpy.bool_)),
                )
            ],
            ["x"],
        ),
    )

    ## record → different tuple
    with pytest.raises(ValueError, match=r"converted between records and tuples"):
        ak.enforce_type(
            ak.to_layout([{"x": [1, 2]}], regulararray=False),
            ak.types.from_datashape("(var * float64)", highlevel=False),
        )

    ## tuple → different record
    with pytest.raises(ValueError, match=r"converted between records and tuples"):
        ak.enforce_type(
            ak.to_layout([([1, 2],)], regulararray=False),
            ak.types.from_datashape("{x: var * float64}", highlevel=False),
        )

    with pytest.raises(
        TypeError, match=r"can only add new slots to a tuple if they are option types"
    ):
        ak.enforce_type(
            ak.to_layout([{"x": [1, 2]}], regulararray=False),
            ak.types.from_datashape("{y: var * float64}", highlevel=False),
        )

    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{y: ?var * float64}", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RecordArray(
            [
                ak.contents.IndexedOptionArray(
                    ak.index.Index64([-1]),
                    ak.contents.ListOffsetArray(
                        ak.index.Index(numpy.array([0], dtype=numpy.int64)),
                        ak.contents.NumpyArray(numpy.array([], dtype=numpy.float64)),
                    ),
                )
            ],
            ["y"],
        ),
    )
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{x: var * int64, y: ?int64}", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                ),
                ak.contents.IndexedOptionArray(
                    ak.index.Index64([-1]),
                    ak.contents.NumpyArray(numpy.array([], dtype=numpy.int64)),
                ),
            ],
            ["x", "y"],
        ),
    )
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{}", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RecordArray([], [], length=1),
    )


def test_list():
    #
    # List types
    result = ak.enforce_type(
        ak.to_layout([[1, 2, 3]]),
        ak.types.from_datashape("var * int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)),
        ),
    )
    result = ak.enforce_type(
        ak.to_layout([[1, 2, 3]]), ak.types.from_datashape("3 * int64", highlevel=False)
    )
    assert ak.almost_equal(
        result,
        ak.contents.RegularArray(
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)), size=3
        ),
    )
    with pytest.raises(ValueError, match=r"converted .* different size"):
        ak.enforce_type(
            ak.to_layout([[1, 2, 3]]),
            ak.types.from_datashape("4 * int64", highlevel=False),
        )

    # Regular types
    result = ak.enforce_type(
        ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False),
        ak.types.from_datashape("var * int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)),
        ),
    )
    result = ak.enforce_type(
        ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False),
        ak.types.from_datashape("3 * int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RegularArray(
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)), size=3
        ),
    )
    with pytest.raises(ValueError, match=r"different size"):
        ak.enforce_type(
            ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False),
            ak.types.from_datashape("4 * int64", highlevel=False),
        )


def test_option():
    # Options
    ## option → option
    result = ak.enforce_type(
        ak.to_layout([1, None]),
        ak.types.from_datashape("?int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.IndexedOptionArray(
            ak.index.Index(numpy.array([0, -1], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
        ),
    )

    ## option → no option
    result = ak.enforce_type(
        ak.to_layout([1, None])[:1],
        ak.types.from_datashape("int64", highlevel=False),
    )
    assert ak.almost_equal(
        result, ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64))
    )

    with pytest.raises(ValueError, match=r"if there are no missing values"):
        ak.enforce_type(
            ak.to_layout([1, None]),
            ak.types.from_datashape("int64", highlevel=False),
        )

    ## Add option
    result = ak.enforce_type(
        ak.to_layout([1, 2]),
        ak.types.from_datashape("?int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.UnmaskedArray(
            ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64))
        ),
    )


def test_numpy():
    ## NumPy
    ## 1D → 1D
    result = ak.enforce_type(
        ak.to_layout([1, 2]),
        ak.types.from_datashape("int64", highlevel=False),
    )
    assert ak.almost_equal(result, numpy.array([1, 2], dtype=numpy.int64))

    result = ak.enforce_type(
        ak.to_layout([1, 2]),
        ak.types.from_datashape("float32", highlevel=False),
    )
    assert ak.almost_equal(result, numpy.array([1.0, 2.0], dtype=numpy.float32))

    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("string", highlevel=False),
        )

    ## 1D → 2D
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("var * int64", highlevel=False),
        )
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("2 * float32", highlevel=False),
        )
    ## 2D → 1D
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
            ak.types.from_datashape("int64", highlevel=False),
        )
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
            ak.types.from_datashape("float32", highlevel=False),
        )

    ## 2D → 2D
    result = ak.enforce_type(
        ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
        ak.types.from_datashape("var * int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3, 6], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([0, 0, 0, 0, 0, 0], dtype=numpy.int64)),
        ),
    )

    result = ak.enforce_type(
        ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
        ak.types.from_datashape("3 * float32", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.RegularArray(
            ak.contents.NumpyArray(
                numpy.array([0, 0, 0, 0, 0, 0], dtype=numpy.float32)
            ),
            size=3,
        ),
    )


def test_union():
    # Unions

    ## non union → union
    result = ak.enforce_type(
        ak.to_layout([1, 2]),
        ak.types.from_datashape("union[int64, string]", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 0]),
            index=ak.index.Index64([0, 1]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0]),
                    content=ak.contents.NumpyArray(
                        numpy.array([], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string"},
                ),
            ],
        ),
    )

    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("union[var * int64, string]", highlevel=False),
        )
    ## union → no union
    result = ak.enforce_type(
        # Build union layout, slice to test projection
        ak.to_layout([1, "hi", "bye"])[1:2],
        ak.types.from_datashape("string", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 2]),
            content=ak.contents.NumpyArray(
                numpy.array([104, 105], dtype=numpy.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    )
    result = ak.enforce_type(
        # Build union layout, slice to test projection
        ak.to_layout([1, "hi", "bye"])[:1],
        ak.types.from_datashape("int64", highlevel=False),
    )
    assert ak.almost_equal(
        result, ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64))
    )

    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, "hi"]),
            ak.types.from_datashape("var * int64", highlevel=False),
        )

    ## union → same union
    result = ak.enforce_type(
        ak.to_layout([1, "hi"]),
        ak.types.from_datashape("union[int64, string]", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0, 2]),
                    content=ak.contents.NumpyArray(
                        numpy.array([104, 105], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string"},
                ),
            ],
        ),
    )

    ## union → bigger union
    result = ak.enforce_type(
        ak.to_layout([1, "hi"]),
        ak.types.from_datashape("union[int64, string, datetime64]", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0, 2]),
                    content=ak.contents.NumpyArray(
                        numpy.array([104, 105], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string"},
                ),
                ak.contents.NumpyArray(numpy.array([], dtype=numpy.datetime64)),
            ],
        ),
    )

    ## union → different union (same N)
    result = ak.enforce_type(
        ak.to_layout([1, "hi"]),
        ak.types.from_datashape("union[float32, string]", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.float32)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0, 2]),
                    content=ak.contents.NumpyArray(
                        numpy.array([104, 105], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string"},
                ),
            ],
        ),
    )

    ## union → incompatible different union (same N)
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, "hi"]),
            ak.types.from_datashape("union[int64, bool]", highlevel=False),
        )

    ## union → different union (same N, more than one change)
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, "hi", False]),
            ak.types.from_datashape(
                "union[datetime64, string, float32]", highlevel=False
            ),
        )


def test_string():
    ## string -> bytestring
    result = ak.enforce_type(
        ak.to_layout(["hello world"]),
        ak.types.from_datashape("bytes", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.uint8,
                ),
                parameters={"__array__": "byte"},
            ),
            parameters={"__array__": "bytestring"},
        ),
    )

    ## bytestring -> string
    result = ak.enforce_type(
        ak.to_layout([b"hello world"]),
        ak.types.from_datashape("string", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.uint8,
                ),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    )

    ## string -> string
    result = ak.enforce_type(
        ak.to_layout(["hello world"]),
        ak.types.from_datashape("string", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.uint8,
                ),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    )

    ## bytestring -> bytestring
    result = ak.enforce_type(
        ak.to_layout([b"hello world"]),
        ak.types.from_datashape("bytes", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.uint8,
                ),
                parameters={"__array__": "byte"},
            ),
            parameters={"__array__": "bytestring"},
        ),
    )

    ## bytestring -> list of byte
    result = ak.enforce_type(
        ak.to_layout([b"hello world"]),
        ak.types.from_datashape("var * byte", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.uint8,
                ),
                parameters={"__array__": "byte"},
            ),
        ),
    )

    ## bytestring -> list of int64
    result = ak.enforce_type(
        ak.to_layout([b"hello world"]),
        ak.types.from_datashape("var * int64", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.int64,
                )
            ),
        ),
    )

    ## list of int64 -> string
    result = ak.enforce_type(
        ak.without_parameters([b"hello world"]),
        ak.types.from_datashape("string", highlevel=False),
    )
    assert ak.almost_equal(
        result,
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 11]),
            content=ak.contents.NumpyArray(
                numpy.array(
                    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                    dtype=numpy.uint8,
                ),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    )
