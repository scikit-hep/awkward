# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# NumPy rejects a zero step in every dimension; Awkward used to do so only for
# the outermost one, and spun forever inside `awkward_ListArray_getitem_next_*`
# for an inner one.

listoffset = ak.contents.ListOffsetArray(
    ak.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
    ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
)

LAYOUTS = {
    "ListArray": ak.contents.ListArray(
        ak.index.Index64(np.array([0, 3], dtype=np.int64)),
        ak.index.Index64(np.array([3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
    ),
    "ListOffsetArray": listoffset,
    "RegularArray": ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)), 3
    ),
    "NumpyArray": ak.contents.NumpyArray(np.arange(6, dtype=np.int64).reshape(2, 3)),
    "ByteMaskedArray": ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([1, 1], dtype=np.int8)), listoffset, valid_when=True
    ),
    "BitMaskedArray": ak.contents.BitMaskedArray(
        ak.index.IndexU8(np.array([0b11], dtype=np.uint8)),
        listoffset,
        valid_when=True,
        length=2,
        lsb_order=True,
    ),
    "UnmaskedArray": ak.contents.UnmaskedArray(listoffset),
    "IndexedArray": ak.contents.IndexedArray(
        ak.index.Index64(np.array([1, 0], dtype=np.int64)), listoffset
    ),
    "IndexedOptionArray": ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([1, -1, 0], dtype=np.int64)), listoffset
    ),
    "RecordArray": ak.contents.RecordArray([listoffset], ["x"]),
    "UnionArray": ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1], dtype=np.int64)),
        [listoffset, listoffset],
    ),
    "EmptyArray": ak.contents.RegularArray(ak.contents.EmptyArray(), 0, 2),
}


@pytest.mark.parametrize("layout", LAYOUTS.values(), ids=LAYOUTS.keys())
@pytest.mark.parametrize(
    "where", [(slice(None, None, 0),), (slice(0, 2), slice(None, None, 0))]
)
@pytest.mark.parametrize("typetracer", [False, True])
def test_zero_step_raises(layout, where, typetracer):
    if typetracer:
        layout = layout.to_typetracer(forget_length=True)
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        layout[where]


@pytest.mark.parametrize(
    "where",
    [
        (slice(None, None, 0),),
        (slice(0, 2), slice(None, None, 0)),
        (slice(None), slice(1, 2, 0)),
        (slice(None), slice(None, None, False)),
        (slice(None), slice(None, None, np.int64(0))),
        (Ellipsis, slice(None, None, 0)),
        (None, slice(None), slice(None, None, 0)),
        (np.array([0, 1]), slice(None, None, 0)),
    ],
)
def test_zero_step_slice_shapes(where):
    array = ak.Array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        array[where]


def test_zero_step_deeper_nesting():
    array = ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        array[0:2, 0:2, ::0]
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        array[:, ::0, :]


def test_zero_step_index():
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        ak.index.Index64(np.arange(5))[::0]


def test_nonzero_step_still_works():
    array = ak.Array([[1, 2, 3], [4, 5, 6]])
    assert array[:, ::2].to_list() == [[1, 3], [4, 6]]
    assert array[:, ::-1].to_list() == [[3, 2, 1], [6, 5, 4]]
    assert array[::-1, 1:3].to_list() == [[5, 6], [2, 3]]

    option = ak.Array([[1, None, 3], [4, 5, None]])
    assert option[:, ::-1].to_list() == [[3, None, 1], [None, 5, 4]]


def test_unknown_typetracer_step_is_not_rejected():
    from awkward._nplikes.shape import unknown_length
    from awkward._nplikes.typetracer import TypeTracerArray

    layout = ak.Array([[1, 2, 3], [4, 5, 6]]).layout.to_typetracer(forget_length=True)
    # Neither an unknown length nor an unknown scalar is *known* to be zero, so
    # the guard must let them through untouched.
    assert layout[:, ::unknown_length].purelist_depth == 2
    unknown_step = TypeTracerArray._new(np.dtype(np.int64), ())
    assert layout[:, ::unknown_step].purelist_depth == 2
