# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward.operations.ak_concatenate import enforce_concatenated_form

layouts = [
    # ListArray
    ak.contents.ListArray(
        ak.index.Index64([0, 3]),
        ak.index.Index64([3, 6]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
    ),
    # ListArray
    ak.contents.ListOffsetArray(
        ak.index.Index64([0, 3, 6]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
    ),
    # RegularArray
    ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(6, dtype=np.int64)), 3),
    ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(6, dtype=np.int64)), 2),
    # ByteMaskedArray
    ak.contents.ByteMaskedArray(
        ak.index.Index8([True, False, False, True]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
        valid_when=True,
    ),
    # ByteMaskedArray
    ak.contents.BitMaskedArray(
        ak.index.IndexU8([1 << 0 | 1 << 1 | 0 << 2 | 0 << 3 | 1 << 4 | 0 << 5]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
        valid_when=True,
        lsb_order=True,
        length=6,
    ),
    # UnmaskedArray
    ak.contents.UnmaskedArray(ak.contents.NumpyArray(np.arange(6, dtype=np.int32))),
    # IndexedOptionArray
    ak.contents.IndexedOptionArray(
        ak.index.Index64([3, 1, -1, -1, 2, 0, -1]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
    ),
    # NumpyArray
    ak.contents.NumpyArray(np.arange(6, dtype=np.int16)),
    ak.contents.NumpyArray(np.arange(6 * 4, dtype=np.float32).reshape(6, 4)),
    # IndexedArray
    ak.contents.IndexedArray(
        ak.index.Index64([3, 1, 1, 0, 2, 0, 0]),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
    ),
    # RecordArray
    ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(6, dtype=np.int16))], ["x"]
    ),
    ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(6, dtype=np.float64))], ["y"]
    ),
    ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(6, dtype=np.float32))], None
    ),
    # UnionArray
    ak.contents.UnionArray(
        ak.index.Index8([0, 0, 1]),
        ak.index.Index64([0, 1, 0]),
        [
            ak.contents.NumpyArray(np.arange(6, dtype=np.int16)),
            ak.contents.RecordArray(
                [ak.contents.NumpyArray(np.arange(6, dtype=np.float32))], None
            ),
        ],
    ),
]


@pytest.mark.parametrize("left", layouts)
@pytest.mark.parametrize("right", layouts)
def test_symmetric(left, right):
    result = ak.concatenate([left, right], axis=0, highlevel=False)
    part_0_result = enforce_concatenated_form(left, result.form)
    assert part_0_result.form == result.form

    part_1_result = enforce_concatenated_form(right, result.form)
    assert part_1_result.form == result.form

    assert part_0_result.to_list() == result[: part_0_result.length].to_list()
    assert part_1_result.to_list() == result[part_0_result.length :].to_list()


@pytest.mark.parametrize(
    "left, right",
    [
        (
            # IndexedOptionArray
            ak.contents.IndexedOptionArray(
                ak.index.Index64([3, 1, -1, -1, 2, 0, -1]),
                ak.contents.NumpyArray(np.arange(6, dtype=np.int32)),
                parameters={"__array__": "categorical"},
            ),
            # NumpyArray
            ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
        ),
    ],
)
def test_non_diagonal(left, right):
    result = ak.concatenate([left, right], axis=0, highlevel=False)
    part_0_result = enforce_concatenated_form(left, result.form)
    assert part_0_result.form == result.form

    part_1_result = enforce_concatenated_form(right, result.form)
    assert part_1_result.form == result.form

    assert part_0_result.to_list() == result[: part_0_result.length].to_list()
    assert part_1_result.to_list() == result[part_0_result.length :].to_list()


# def test_union_
