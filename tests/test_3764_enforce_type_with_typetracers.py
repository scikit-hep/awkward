# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import TypeTracerArray

# tests taken from tests/test_2365_enforce_type.py and adapted for typetracer arrays


@pytest.mark.parametrize("forget_length", [False, True])
def test_record(forget_length):
    ## record → record
    original = ak.to_layout([{"x": [1, 2]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("{x: var * int64}", highlevel=False)
    )
    expected = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
            )
        ],
        ["x"],
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## record → record
    original = ak.to_layout([{"x": [1, 0]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("{x: var * bool}", highlevel=False)
    )
    expected = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                ak.contents.NumpyArray(numpy.array([1, 0], dtype=numpy.bool_)),
            )
        ],
        ["x"],
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## record → different tuple
    original = ak.to_layout([{"x": [1, 2]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(ValueError, match=r"converted between records and tuples"):
        ak.enforce_type(
            array, ak.types.from_datashape("(var * float64)", highlevel=False)
        )

    original = ak.to_layout([{"x": [1, 2]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(
        TypeError, match=r"can only add new fields to a record if they are option types"
    ):
        ak.enforce_type(
            array, ak.types.from_datashape("{y: var * float64}", highlevel=False)
        )

    ## record → totally different record
    original = ak.to_layout([{"x": [1, 2]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("{y: ?var * float64}", highlevel=False)
    )
    expected = ak.contents.RecordArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## record → extended record
    original = ak.to_layout([{"x": [1, 2]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
        ak.types.from_datashape("{x: var * int64, y: ?int64}", highlevel=False),
    )
    expected = ak.contents.RecordArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## record → empty record
    original = ak.to_layout([{"x": [1, 2]}], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("{}", highlevel=False))
    expected = ak.contents.RecordArray([], [], length=1).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ############

    ## tuple → tuple
    original = ak.to_layout([([1, 2],)], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("(var * int64)", highlevel=False)
    )
    expected = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64)),
            )
        ],
        None,
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## tuple → tuple
    original = ak.to_layout([([1, 0],)], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("(var * bool)", highlevel=False)
    )
    expected = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                ak.contents.NumpyArray(numpy.array([1, 0], dtype=numpy.bool_)),
            )
        ],
        None,
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## tuple → different record
    original = ak.to_layout([([1, 2],)], regulararray=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(ValueError, match=r"converted between records and tuples"):
        ak.enforce_type(
            array, ak.types.from_datashape("{x: var * float64}", highlevel=False)
        )

    original = ak.to_layout([([1, 2],)], regulararray=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(
        TypeError, match=r"can only add new slots to a tuple if they are option types"
    ):
        ak.enforce_type(
            array, ak.types.from_datashape("(var * int64, float32)", highlevel=False)
        )

    ## tuple → extended tuple
    original = ak.to_layout([([1, 2],)], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("(var * int64, ?float32)", highlevel=False)
    )
    expected = ak.contents.RecordArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## tuple → empty tuple
    original = ak.to_layout([([1, 2],)], regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("()", highlevel=False))
    expected = ak.contents.RecordArray([], None, length=1).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_list(forget_length):
    original = ak.to_layout([[1, 2, 3]])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("var * int64", highlevel=False)
    )
    expected = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([[1, 2, 3]])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("3 * int64", highlevel=False)
    )
    expected = ak.contents.RegularArray(
        ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)), size=3
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## Empty list to regular shape
    original = ak.to_layout([[]])[:0]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("3 * int64", highlevel=False)
    )
    expected = ak.contents.RegularArray(
        ak.contents.NumpyArray(numpy.array([], dtype=numpy.int64)), size=3
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([[1, 2, 3]])
    array = original.to_typetracer(forget_length)
    expected = ak.contents.RegularArray(
        ak.contents.NumpyArray(
            TypeTracerArray._new(numpy.dtype("int64"), (unknown_length,))
        ),
        size=unknown_length,
    )
    assert result.layout.is_equal_to(expected)

    original = ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("var * int64", highlevel=False)
    )
    expected = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("3 * int64", highlevel=False)
    )
    expected = ak.contents.RegularArray(
        ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64)), size=3
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_regular([[1, 2, 3]], axis=-1, highlevel=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(ValueError, match=r"different size"):
        ak.enforce_type(array, ak.types.from_datashape("4 * int64", highlevel=False))


@pytest.mark.parametrize("forget_length", [False, True])
def test_option(forget_length):
    ## option → option
    original = ak.to_layout([1, None])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("?int64", highlevel=False))
    expected = ak.contents.IndexedOptionArray(
        ak.index.Index(numpy.array([0, -1], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## option → option (packing)
    original = ak.to_layout([1, None, 2, 3])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("?float64", highlevel=False)
    )
    expected = ak.contents.IndexedOptionArray(
        ak.index.Index(numpy.array([0, -1, 1, 2], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.float64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## option → no option
    original = ak.to_layout([1, None])[:1]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("int64", highlevel=False))
    expected = ak.contents.IndexedArray(
        ak.index.Index(numpy.array([0], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, None])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("int64", highlevel=False))
    expected = ak.contents.IndexedArray(
        ak.index.Index(numpy.array([0], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## Add option
    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("?int64", highlevel=False))
    expected = ak.contents.UnmaskedArray(
        ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.int64))
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## option[X] → option[unknown]
    original = ak.to_layout([None, 1, 2, 3])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, "?unknown")
    expected = ak.contents.IndexedOptionArray(
        ak.index.Index64([-1, -1, -1, -1]), ak.contents.EmptyArray()
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_numpy(forget_length):
    ## NumPy
    ## 1D → 1D
    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("int64", highlevel=False))
    expected = ak.contents.NumpyArray(
        numpy.array([1, 2], dtype=numpy.int64)
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("float32", highlevel=False))
    expected = ak.contents.NumpyArray(
        numpy.array([1.0, 2.0], dtype=numpy.float32)
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(array, ak.types.from_datashape("string", highlevel=False))

    ## 1D → 2D
    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(array, ak.types.from_datashape("var * int64", highlevel=False))

    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(array, ak.types.from_datashape("2 * float32", highlevel=False))

    ## 2D → 1D
    original = ak.to_layout(numpy.zeros((2, 3)), regulararray=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(array, ak.types.from_datashape("int64", highlevel=False))

    original = ak.to_layout(numpy.zeros((2, 3)), regulararray=False)
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(array, ak.types.from_datashape("float32", highlevel=False))

    ## 2D → 2D
    original = ak.to_layout(numpy.zeros((2, 3)), regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("var * int64", highlevel=False)
    )
    expected = ak.contents.ListOffsetArray(
        ak.index.Index(numpy.array([0, 3, 6], dtype=numpy.int64)),
        ak.contents.NumpyArray(numpy.array([0, 0, 0, 0, 0, 0], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout(numpy.zeros((2, 3)), regulararray=False)
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("3 * float32", highlevel=False)
    )
    expected = ak.contents.RegularArray(
        ak.contents.NumpyArray(numpy.array([0, 0, 0, 0, 0, 0], dtype=numpy.float32)),
        size=3,
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_union(forget_length):
    ## non union → union
    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("union[int64, string]", highlevel=False)
    )
    expected = ak.contents.UnionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, 2])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(
            array,
            ak.types.from_datashape("union[var * int64, string]", highlevel=False),
        )

    ## union → no union (project)
    original = ak.to_layout([1, "hi", "bye"])[1:2]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("string", highlevel=False))
    expected = ak.contents.IndexedArray(
        ak.index.Index64([0]),
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 2, 5]),
            content=ak.contents.NumpyArray(
                numpy.array([104, 105, 98, 121, 101], dtype=numpy.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, "hi", "bye"])[:1]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("int64", highlevel=False))
    expected = ak.contents.IndexedArray(
        ak.index.Index64([0]),
        ak.contents.NumpyArray(numpy.array([1], dtype=numpy.int64)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, "hi"])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(array, ak.types.from_datashape("var * int64", highlevel=False))

    ## union → no union (convert)
    original = ak.concatenate(
        [
            [{"x": 1, "y": 2.0}],
            [{"y": 3}],
            [{"x": 4, "y": 5.0}],
            [{"y": 6}],
        ],
    )
    array = original.layout.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("{y: int64}", highlevel=False)
    )
    expected = ak.contents.IndexedArray(
        index=ak.index.Index64([0, 2, 1, 3]),
        content=ak.contents.RecordArray(
            contents=[
                ak.contents.NumpyArray(
                    numpy.array([2, 5, 3, 6], dtype=numpy.int64),
                )
            ],
            fields=["y"],
        ),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.to_layout([1, "hi"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("string", highlevel=False))
    expected = ak.contents.IndexedArray(
        ak.index.Index64([0]),
        ak.contents.ListOffsetArray(
            offsets=ak.index.Index64([0, 2]),
            content=ak.contents.NumpyArray(
                numpy.array([104, 105], dtype=numpy.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## union → same union
    original = ak.to_layout([1, "hi"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
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
    expected = ak.contents.UnionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## union → bigger union
    original = ak.to_layout([1, "hi"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
        ak.types.from_datashape("union[int64, string, datetime64]", highlevel=False),
    )
    expected = ak.contents.UnionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## union → different union (same N)
    original = ak.to_layout([1, "hi"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
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
    expected = ak.contents.UnionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## union → different union (smaller N)
    original = ak.to_layout([1, "hi", [1j, 2j]])[:2]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, "union[int64, string]")
    expected = ak.contents.UnionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## union → incompatible different union (same N)
    original = ak.to_layout([1, "hi"])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(
            array, ak.types.from_datashape("union[int64, bool]", highlevel=False)
        )

    ## union → different union (same N, more than one change)
    original = ak.to_layout([1, "hi", False])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError):
        ak.enforce_type(
            array,
            ak.types.from_datashape(
                "union[datetime64, string, float32]", highlevel=False
            ),
        )

    ## union of union → union of extended union
    original = ak.contents.UnionArray(
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
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
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
    expected = ak.contents.UnionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_string(forget_length):
    ## string -> bytestring
    original = ak.to_layout(["hello world"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("bytes", highlevel=False))
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.uint8,
            ),
            parameters={"__array__": "byte"},
        ),
        parameters={"__array__": "bytestring"},
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## bytestring -> string
    original = ak.to_layout([b"hello world"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("string", highlevel=False))
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.uint8,
            ),
            parameters={"__array__": "char"},
        ),
        parameters={"__array__": "string"},
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## string -> string
    original = ak.to_layout(["hello world"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("string", highlevel=False))
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.uint8,
            ),
            parameters={"__array__": "char"},
        ),
        parameters={"__array__": "string"},
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## bytestring -> bytestring
    original = ak.to_layout([b"hello world"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("bytes", highlevel=False))
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.uint8,
            ),
            parameters={"__array__": "byte"},
        ),
        parameters={"__array__": "bytestring"},
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## bytestring -> list of byte
    original = ak.to_layout([b"hello world"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("var * byte", highlevel=False)
    )
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.uint8,
            ),
            parameters={"__array__": "byte"},
        ),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## bytestring -> list of int64
    original = ak.to_layout([b"hello world"])
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("var * int64", highlevel=False)
    )
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.int64,
            )
        ),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## list of int64 -> string
    original = ak.without_parameters([b"hello world"])
    array = original.layout.to_typetracer(forget_length)
    result = ak.enforce_type(array, ak.types.from_datashape("string", highlevel=False))
    expected = ak.contents.ListOffsetArray(
        offsets=ak.index.Index64([0, 11]),
        content=ak.contents.NumpyArray(
            numpy.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100],
                dtype=numpy.uint8,
            ),
            parameters={"__array__": "char"},
        ),
        parameters={"__array__": "string"},
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_highlevel(forget_length):
    original = ak.to_layout(["hello world"])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError, match=r"High-level type objects are not supported"):
        ak.enforce_type(array, ak.types.from_datashape("1 * bytes"))

    original = ak.to_layout([{"msg": "hello world"}])
    array = original.to_typetracer(forget_length)
    with pytest.raises(TypeError, match=r"High-level type objects are not supported"):
        ak.enforce_type(
            array,
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


@pytest.mark.parametrize("forget_length", [False, True])
def test_single_record(forget_length):
    original = ak.to_regular([{"x": [1, 2]}])
    record = original.layout.to_typetracer(forget_length)[0]
    result = ak.enforce_type(
        record,
        ak.types.from_datashape("{x: var * float64}", highlevel=False),
        highlevel=False,
    )
    expected = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index(numpy.array([0, 2], dtype=numpy.int64)),
                ak.contents.NumpyArray(numpy.array([1, 2], dtype=numpy.float64)),
            )
        ],
        ["x"],
    ).to_typetracer(forget_length)
    assert result.array.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_indexed(forget_length):
    original = ak.contents.IndexedArray(
        ak.index.Index64([0, 2]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 6, 9]),
            ak.contents.NumpyArray(numpy.arange(9, dtype=numpy.int64)),
        ),
    )
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
        ak.types.from_datashape(
            'var * int64[parameters={"key": "value"}]', highlevel=False
        ),
    )
    expected = ak.contents.IndexedArray(
        ak.index.Index64([0, 2]),
        ak.contents.ListOffsetArray(
            ak.index.Index(numpy.array([0, 3, 6, 9], dtype=numpy.int64)),
            ak.contents.NumpyArray(
                numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=numpy.int64),
                parameters={"key": "value"},
            ),
        ),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.contents.IndexedArray(
        ak.index.Index64([0, 2]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 6, 9]),
            ak.contents.NumpyArray(numpy.arange(9, dtype=numpy.int64)),
        ),
    )
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("var * float32", highlevel=False)
    )
    expected = ak.contents.ListOffsetArray(
        ak.index.Index64([0, 3, 6]),
        ak.contents.NumpyArray(numpy.array([0, 1, 2, 6, 7, 8], dtype=numpy.float32)),
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_unknown(forget_length):
    original = ak.contents.EmptyArray()
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, "unknown")
    expected = ak.contents.EmptyArray().to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    original = ak.contents.EmptyArray()
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, "int64")
    expected = ak.contents.NumpyArray(numpy.empty(0, numpy.int64)).to_typetracer(
        forget_length
    )
    assert result.layout.is_equal_to(expected)

    original = ak.contents.NumpyArray(numpy.empty(0, numpy.int64))
    array = original.to_typetracer(forget_length)
    with pytest.raises(
        TypeError, match=r"cannot convert non-EmptyArray layouts to a bare UnknownType"
    ):
        ak.enforce_type(array, "unknown")

    original = ak.contents.NumpyArray(numpy.empty(0, numpy.int64))
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(array, "?unknown")
    expected = ak.contents.IndexedOptionArray(
        ak.index.Index64([]), ak.contents.EmptyArray()
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)


@pytest.mark.parametrize("forget_length", [False, True])
def test_misc(forget_length):
    ## option → option (inside indexed)
    original = ak.to_layout([{"x": [1, 2, None]}, None, {"x": [3, 4, None]}])[
        [0, 2], :2
    ]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
        """
    ?{
        x: var * ?float32
    }
        """,
    )
    expected = ak.contents.IndexedOptionArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## no option → option (inside indexed)
    original = ak.to_layout(
        [
            {"x": [1, 2, None]},
            {"x": [9, 9, None]},
            {"x": [3, 4, None]},
            {"x": [8, 8, None]},
        ]
    )[[0, 2], :2]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array,
        """
    ?{
        x: var * ?float32
    }
        """,
    )
    expected = ak.contents.UnmaskedArray(
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)

    ## option (indexed) list of union → option list of no union (project)
    original = ak.to_layout([[1, "hi", "bye"], None])[[0, 1], 1:2]
    array = original.to_typetracer(forget_length)
    result = ak.enforce_type(
        array, ak.types.from_datashape("?var * string", highlevel=False)
    )
    expected = ak.contents.IndexedOptionArray(
        ak.index.Index64([0, -1]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 1]),
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
    ).to_typetracer(forget_length)
    assert result.layout.is_equal_to(expected)
