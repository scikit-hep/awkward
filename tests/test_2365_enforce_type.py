# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest

import awkward as ak


def test_record():
    ## record → record
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{x: var * int64}", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                )
            ],
            ["x"],
        )
    )

    ## record → record
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 0]}], regulararray=False),
        ak.types.from_datashape("{x: var * bool}", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 0], dtype=numpy.bool_)),
                )
            ],
            ["x"],
        )
    )

    ## record → different tuple
    with pytest.raises(ValueError, match=r"converted between records and tuples"):
        ak.enforce_type(
            ak.to_layout([{"x": [1, 2]}], regulararray=False),
            ak.types.from_datashape("(var * float64)", highlevel=False),
        )

    with pytest.raises(
        TypeError, match=r"can only add new fields to a record if they are option types"
    ):
        ak.enforce_type(
            ak.to_layout([{"x": [1, 2]}], regulararray=False),
            ak.types.from_datashape("{y: var * float64}", highlevel=False),
        )

    ## record → totally different record
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{y: ?var * float64}", highlevel=False),
    )
    assert result.layout.is_equal_to(
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
        )
    )
    ## record → extended record
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{x: var * int64, y: ?int64}", highlevel=False),
    )
    assert result.layout.is_equal_to(
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
    ## record → empty record
    result = ak.enforce_type(
        ak.to_layout([{"x": [1, 2]}], regulararray=False),
        ak.types.from_datashape("{}", highlevel=False),
    )
    assert result.layout.is_equal_to(ak.contents.RecordArray([], [], length=1))

    ############

    ## tuple → tuple
    result = ak.enforce_type(
        ak.to_layout([([1, 2],)], regulararray=False),
        ak.types.from_datashape("(var * int64)", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                )
            ],
            None,
        )
    )

    ## tuple → tuple
    result = ak.enforce_type(
        ak.to_layout([([1, 0],)], regulararray=False),
        ak.types.from_datashape("(var * bool)", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 0], dtype=numpy.bool_)),
                )
            ],
            None,
        )
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
            ak.to_layout([([1, 2],)], regulararray=False),
            ak.types.from_datashape("(var * int64, float32)", highlevel=False),
        )

    ## tuple → extended tuple
    result = ak.enforce_type(
        ak.to_layout([([1, 2],)], regulararray=False),
        ak.types.from_datashape("(var * int64, ?float32)", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
                ),
                ak.contents.IndexedOptionArray(
                    ak.index.Index64([-1]),
                    ak.contents.NumpyArray(numpy.array([], dtype=numpy.float32)),
                ),
            ],
            None,
        )
    )
    ## tuple → empty tuple
    result = ak.enforce_type(
        ak.to_layout([([1, 2],)], regulararray=False),
        ak.types.from_datashape("()", highlevel=False),
    )
    assert result.layout.is_equal_to(ak.contents.RecordArray([], None, length=1))


def test_list():
    #
    # List types
    result = ak.enforce_type(
        ak.to_layout([[1, 2, 3]]),
        ak.types.from_datashape("var * int64", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)),
        ),
    )
    result = ak.enforce_type(
        ak.to_layout([[1, 2, 3]]), ak.types.from_datashape("3 * int64", highlevel=False)
    )
    assert result.layout.is_equal_to(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)), size=3
        ),
    )
    ## Empty list to regular shape
    result = ak.enforce_type(
        ak.to_layout([[]])[:0], ak.types.from_datashape("3 * int64", highlevel=False)
    )
    assert result.layout.is_equal_to(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(numpy.array([], dtype=numpy.int64)), size=3
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
    assert result.layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)),
        ),
    )
    result = ak.enforce_type(
        ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False),
        ak.types.from_datashape("3 * int64", highlevel=False),
    )
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
        ak.contents.IndexedOptionArray(
            ak.index.Index(numpy.array([0, -1], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
        ),
    )
    ## option → option (packing)
    result = ak.enforce_type(
        ak.to_layout([1, None, 2, 3]),
        ak.types.from_datashape("?float64", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedOptionArray(
            ak.index.Index(numpy.array([0, -1, 1, 2], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.float64)),
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
    assert result.layout.is_equal_to(
        ak.contents.UnmaskedArray(
            ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64))
        ),
    )

    ## option[X] → option[unknown]
    layout = ak.to_layout([None, 1, 2, 3])
    result = ak.enforce_type(layout, "?unknown")
    assert result.layout.is_equal_to(
        ak.contents.IndexedOptionArray(
            ak.index.Index64([-1, -1, -1, -1]), ak.contents.EmptyArray()
        )
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

    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("string", highlevel=False),
        )

    ## 1D → 2D
    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("var * int64", highlevel=False),
        )
    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("2 * float32", highlevel=False),
        )
    ## 2D → 1D
    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
            ak.types.from_datashape("int64", highlevel=False),
        )
    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
            ak.types.from_datashape("float32", highlevel=False),
        )

    ## 2D → 2D
    result = ak.enforce_type(
        ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
        ak.types.from_datashape("var * int64", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3, 6], dtype=numpy.int64)),
            ak.contents.NumpyArray(numpy.array([0, 0, 0, 0, 0, 0], dtype=numpy.int64)),
        ),
    )

    result = ak.enforce_type(
        ak.to_layout(numpy.zeros((2, 3)), regulararray=False),
        ak.types.from_datashape("3 * float32", highlevel=False),
    )
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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
        )
    )

    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, 2]),
            ak.types.from_datashape("union[var * int64, string]", highlevel=False),
        )

    ## union → no union (project)
    result = ak.enforce_type(
        # Build union layout, slice to test projection
        ak.to_layout([1, "hi", "bye"])[1:2],
        ak.types.from_datashape("string", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedArray(
            ak.index.Index64([0]),
            ak.contents.ListOffsetArray(
                offsets=ak.index.Index64([0, 2, 5]),
                content=ak.contents.NumpyArray(
                    numpy.array([104, 105, 98, 121, 101], dtype=numpy.uint8),
                    parameters={"__array__": "char"},
                ),
                parameters={"__array__": "string"},
            ),
        )
    )
    result = ak.enforce_type(
        # Build union layout, slice to test projection
        ak.to_layout([1, "hi", "bye"])[:1],
        ak.types.from_datashape("int64", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedArray(
            ak.index.Index64([0]),
            ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
        )
    )

    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, "hi"]),
            ak.types.from_datashape("var * int64", highlevel=False),
        )

    ## union → no union (convert)
    array = ak.concatenate(
        [
            # {x: int64, y: float64}
            [{"x": 1, "y": 2.0}],
            # {y: int64}
            [{"y": 3}],
            # {x: int64, y: float64}
            [{"x": 4, "y": 5.0}],
            # {y: int64}
            [{"y": 6}],
        ],
    )
    assert array.type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.RecordType(
                    [ak.types.NumpyType("int64"), ak.types.NumpyType("float64")],
                    ["x", "y"],
                ),
                ak.types.RecordType([ak.types.NumpyType("int64")], ["y"]),
            ]
        ),
        4,
    )
    result = ak.enforce_type(
        array, ak.types.from_datashape("{y: int64}", highlevel=False)
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedArray(
            index=ak.index.Index64([0, 2, 1, 3]),
            content=ak.contents.RecordArray(
                contents=[
                    ak.contents.NumpyArray(
                        numpy.array([2, 5, 3, 6], dtype=numpy.int64),
                    )
                ],
                fields=["y"],
            ),
        )
    )
    with pytest.raises(ValueError):
        ak.enforce_type(
            ak.to_layout([1, "hi"]), ak.types.from_datashape("string", highlevel=False)
        )

    ## union → same union
    result = ak.enforce_type(
        ak.to_layout([1, "hi"]),
        ak.types.UnionType(
            [
                ak.types.NumpyType("int64"),
                ak.types.ListType(
                    ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                    parameters={"__array__": "string", "foo": "bar"},
                ),
            ]
        ),
    )
    assert result.layout.is_equal_to(
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0, 2]),
                    content=ak.contents.NumpyArray(
                        numpy.array([104, 105], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string", "foo": "bar"},
                ),
            ],
        )
    )

    ## union → bigger union
    result = ak.enforce_type(
        ak.to_layout([1, "hi"]),
        ak.types.from_datashape("union[int64, string, datetime64]", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
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
        )
    )

    ## union → different union (same N)
    result = ak.enforce_type(
        ak.to_layout([1, "hi"]),
        ak.types.UnionType(
            [
                ak.types.NumpyType("float32"),
                ak.types.ListType(
                    ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                    parameters={"__array__": "string", "foo": "bar"},
                ),
            ]
        ),
    )
    assert result.layout.is_equal_to(
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1], dtype=numpy.float32)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0, 2]),
                    content=ak.contents.NumpyArray(
                        numpy.array([104, 105], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string", "foo": "bar"},
                ),
            ],
        )
    )

    ## union → different union (smaller N)
    result = ak.enforce_type(
        ak.to_layout([1, "hi", [1j, 2j]])[:2],
        "union[int64, string]",
    )
    assert result.layout.is_equal_to(
        ak.contents.UnionArray(
            tags=ak.index.Index8([0, 1]),
            index=ak.index.Index64([0, 0]),
            contents=[
                ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
                ak.contents.ListOffsetArray(
                    offsets=ak.index.Index64([0, 2]),
                    content=ak.contents.NumpyArray(
                        numpy.array([104, 105], dtype=numpy.uint8),
                        parameters={"__array__": "char"},
                    ),
                    parameters={"__array__": "string"},
                ),
            ],
        )
    )

    ## union → incompatible different union (same N)
    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, "hi"]),
            ak.types.from_datashape("union[int64, bool]", highlevel=False),
        )

    ## union → different union (same N, more than one change)
    with pytest.raises(TypeError):
        ak.enforce_type(
            ak.to_layout([1, "hi", False]),
            ak.types.from_datashape(
                "union[datetime64, string, float32]", highlevel=False
            ),
        )

    ## union of union → union of extended union
    layout = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 2]),
                ak.contents.UnionArray(
                    ak.index.Index8([0, 1]),
                    ak.index.Index64([0, 0]),
                    [
                        ak.contents.NumpyArray(numpy.array([2], dtype=numpy.int64)),
                        ak.contents.ListOffsetArray(
                            ak.index.Index64([0, 2]),
                            ak.contents.RecordArray(
                                [
                                    ak.contents.IndexedOptionArray(
                                        ak.index.Index64([0, -1]),
                                        ak.contents.NumpyArray(
                                            numpy.array([1], dtype=numpy.int64)
                                        ),
                                    ),
                                    ak.contents.IndexedOptionArray(
                                        ak.index.Index64([-1, 0]),
                                        ak.contents.NumpyArray(
                                            numpy.array([2], dtype=numpy.int64)
                                        ),
                                    ),
                                ],
                                ["x", "y"],
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )
    result = ak.enforce_type(
        layout,
        """
    union[
        int64,
        var * union[
            int64,
            var * {
                x: ?int64,
                y: ?int64,
                z: ?string
            }
        ]
    ]
                             """,
    )
    assert result.layout.is_equal_to(
        ak.contents.UnionArray(
            ak.index.Index8([0, 1]),
            ak.index.Index64([0, 0]),
            [
                ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
                ak.contents.ListOffsetArray(
                    ak.index.Index64([0, 2]),
                    ak.contents.UnionArray(
                        ak.index.Index8([0, 1]),
                        ak.index.Index64([0, 0]),
                        [
                            ak.contents.NumpyArray(numpy.array([2], dtype=numpy.int64)),
                            ak.contents.ListOffsetArray(
                                ak.index.Index64([0, 2]),
                                ak.contents.RecordArray(
                                    [
                                        ak.contents.IndexedOptionArray(
                                            ak.index.Index64([0, -1]),
                                            ak.contents.NumpyArray(
                                                numpy.array([1], dtype=numpy.int64)
                                            ),
                                        ),
                                        ak.contents.IndexedOptionArray(
                                            ak.index.Index64([-1, 0]),
                                            ak.contents.NumpyArray(
                                                numpy.array([2], dtype=numpy.int64)
                                            ),
                                        ),
                                        ak.contents.IndexedOptionArray(
                                            ak.index.Index64([-1, -1]),
                                            ak.contents.ListOffsetArray(
                                                ak.index.Index64([0]),
                                                ak.contents.NumpyArray(
                                                    numpy.empty(0, dtype=numpy.uint8),
                                                    parameters={"__array__": "char"},
                                                ),
                                                parameters={"__array__": "string"},
                                            ),
                                        ),
                                    ],
                                    ["x", "y", "z"],
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        )
    )


def test_string():
    ## string -> bytestring
    result = ak.enforce_type(
        ak.to_layout(["hello world"]),
        ak.types.from_datashape("bytes", highlevel=False),
    )
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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
    assert result.layout.is_equal_to(
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


def test_highlevel():
    with pytest.raises(TypeError, match=r"High-level type objects are not supported"):
        ak.enforce_type(
            ak.to_layout(["hello world"]),
            ak.types.from_datashape("1 * bytes"),
        )
    with pytest.raises(TypeError, match=r"High-level type objects are not supported"):
        ak.enforce_type(
            ak.to_layout([{"msg": "hello world"}]),
            ak.types.ScalarType(
                ak.types.RecordType(
                    [
                        ak.types.ListType(
                            ak.types.NumpyType(
                                "uint8", parameters={"__array__": "char"}
                            ),
                            parameters={"__array__": "string"},
                        )
                    ],
                    ["msg"],
                )
            ),
        )


def test_single_record():
    result = ak.enforce_type(
        ak.Record({"x": [1, 2]}),
        ak.types.from_datashape("{x: var * float64}", highlevel=False),
        highlevel=False,
    )
    assert isinstance(result, ak.record.Record)
    assert ak.almost_equal(
        result.array,
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                    ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.float64)),
                )
            ],
            ["x"],
        ),
    )


def test_indexed():
    # Non-packing (because dtype hasn't changed)
    result = ak.enforce_type(
        ak.contents.IndexedArray(
            ak.index.Index64([0, 2]),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 3, 6, 9]),
                ak.contents.NumpyArray(numpy.arange(9, dtype=numpy.int64)),
            ),
        ),
        ak.types.from_datashape(
            'var * int64[parameters={"key": "value"}]', highlevel=False
        ),
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedArray(
            ak.index.Index64([0, 2]),
            ak.contents.ListOffsetArray(
                ak.index.Index(numpy.array([0, 3, 6, 9], dtype=numpy.int64)),
                ak.contents.NumpyArray(
                    numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=numpy.int64),
                    parameters={"key": "value"},
                ),
            ),
        )
    )
    # Packing
    result = ak.enforce_type(
        ak.contents.IndexedArray(
            ak.index.Index64([0, 2]),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 3, 6, 9]),
                ak.contents.NumpyArray(numpy.arange(9, dtype=numpy.int64)),
            ),
        ),
        ak.types.from_datashape("var * float32", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 6]),
            ak.contents.NumpyArray(
                numpy.array([0, 1, 2, 6, 7, 8], dtype=numpy.float32)
            ),
        )
    )


def test_unknown():
    # unknown → unknown
    layout = ak.contents.EmptyArray()
    assert ak.enforce_type(layout, "unknown").layout.is_equal_to(layout)

    # unknown → other
    layout = ak.contents.EmptyArray()
    assert ak.enforce_type(layout, "int64").layout.is_equal_to(
        ak.contents.NumpyArray(numpy.empty(0, numpy.int64))
    )

    # other → unknown
    with pytest.raises(
        TypeError, match=r"cannot convert non-EmptyArray layouts to a bare UnknownType"
    ):
        layout = ak.contents.NumpyArray(numpy.empty(0, numpy.int64))
        ak.enforce_type(layout, "unknown")

    # unknown → other
    layout = ak.contents.NumpyArray(numpy.empty(0, numpy.int64))
    assert ak.enforce_type(layout, "?unknown").layout.is_equal_to(
        ak.contents.IndexedOptionArray(ak.index.Index64([]), ak.contents.EmptyArray())
    )


def test_misc():
    # These tests ensure good coverage over our helper functions
    ## option → option (inside indexed)
    layout = ak.to_layout([{"x": [1, 2, None]}, None, {"x": [3, 4, None]}])[[0, 2], :2]
    result = ak.enforce_type(
        layout,
        """
    ?{
        x: var * ?float32
    }
        """,
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedOptionArray(
            ak.index.Index64([0, 1]),
            ak.contents.RecordArray(
                [
                    ak.contents.ListOffsetArray(
                        ak.index.Index64([0, 2, 4]),
                        ak.contents.IndexedOptionArray(
                            ak.index.Index64([0, 1, 2, 3]),
                            ak.contents.NumpyArray(
                                numpy.array([1, 2, 3, 4], dtype=numpy.float32)
                            ),
                        ),
                    )
                ],
                ["x"],
            ),
        )
    )
    ## no option → option (inside indexed)
    layout = ak.to_layout(
        [
            {"x": [1, 2, None]},
            {"x": [9, 9, None]},
            {"x": [3, 4, None]},
            {"x": [8, 8, None]},
        ]
    )[[0, 2], :2]
    result = ak.enforce_type(
        layout,
        """
    ?{
        x: var * ?float32
    }
        """,
    )
    assert result.layout.is_equal_to(
        ak.contents.UnmaskedArray(
            ak.contents.RecordArray(
                [
                    ak.contents.ListOffsetArray(
                        ak.index.Index64([0, 2, 4]),
                        ak.contents.IndexedOptionArray(
                            ak.index.Index64([0, 1, 2, 3]),
                            ak.contents.NumpyArray(
                                numpy.array([1, 2, 3, 4], dtype=numpy.float32)
                            ),
                        ),
                    )
                ],
                ["x"],
            ),
        )
    )

    ## option (indexed) list of union → option list of no union (project)
    layout = ak.to_layout([[1, "hi", "bye"], None])[[0, 1], 1:2]
    assert isinstance(layout, ak.contents.IndexedOptionArray)
    result = ak.enforce_type(
        # Build union layout, slice to test projection/no-projection
        # wrap union in outer option-of-list, and index it to produce a IndexedOptionArray (to test for packing)
        layout,
        ak.types.from_datashape("?var * string", highlevel=False),
    )
    assert result.layout.is_equal_to(
        ak.contents.IndexedOptionArray(
            ak.index.Index64([0, -1]),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 1]),
                # Indexed type because the string dtype doesn't change, so we don't need to back below this point
                ak.contents.IndexedArray(
                    ak.index.Index64([0]),
                    ak.contents.ListOffsetArray(
                        offsets=ak.index.Index64([0, 2, 5]),
                        content=ak.contents.NumpyArray(
                            numpy.array([104, 105, 98, 121, 101], dtype=numpy.uint8),
                            parameters={"__array__": "char"},
                        ),
                        parameters={"__array__": "string"},
                    ),
                ),
            ),
        ),
    )
