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
