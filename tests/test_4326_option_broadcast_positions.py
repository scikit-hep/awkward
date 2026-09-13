# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

INDEX = {np.int32: ak.index.Index32, np.int64: ak.index.Index64}


def option(index, content, parameters=None, dtype=np.int64):
    return ak.contents.IndexedOptionArray(
        INDEX[dtype](np.asarray(index, dtype=dtype)),
        ak.contents.NumpyArray(np.asarray(content, dtype=np.float64)),
        parameters=parameters,
    )


def bytemasked(mask, content):
    return ak.contents.ByteMaskedArray(
        ak.index.Index8(np.asarray(mask, dtype=np.int8)),
        ak.contents.NumpyArray(np.asarray(content, dtype=np.float64)),
        valid_when=True,
    )


def sum_leaves(layouts, **kwargs):
    if all(isinstance(x, ak.contents.NumpyArray) for x in layouts):
        return ak.contents.NumpyArray(sum(x.data for x in layouts))


def expected_sum(*lists):
    """Element-wise sum, None wherever any input is None, recursing into nested lists."""
    if any(isinstance(x, list) for x in lists):
        return [expected_sum(*items) for items in zip(*lists, strict=True)]
    if any(x is None for x in lists):
        return None
    return sum(lists)


# the index of the second array runs out of order and leaves content entries unreferenced,
# so a projection that carried positions instead of index values would scramble it
CASES = {
    "identical": lambda: (
        [ak.Array(option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0]))] * 3
    ),
    "permuted-partial-index": lambda: [
        ak.Array(option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0])),
        ak.Array(
            option([4, -1, 2, 0, -1, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        ),
    ],
    "index32-with-index64": lambda: [
        ak.Array(option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0], dtype=np.int32)),
        ak.Array(
            option([4, -1, 2, 0, -1, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        ),
    ],
    "different-masks": lambda: [
        ak.Array(option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0])),
        ak.Array(option([-1, 0, 1, -1, 2, 3], [10.0, 11.0, 12.0, 13.0])),
    ],
    "option-with-plain": lambda: [
        ak.Array(option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0])),
        ak.Array(np.arange(6, dtype=np.float64)),
    ],
    "bytemasked-with-indexed": lambda: [
        ak.Array(bytemasked([1, 0, 1, 1, 0, 1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ak.Array(
            option([4, -1, 2, 0, -1, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        ),
    ],
    "all-none": lambda: [
        ak.Array(option([-1, -1, -1], [0.0])),
        ak.Array(np.arange(3, dtype=np.float64)),
    ],
    "empty": lambda: [ak.Array(option([], [0.0])), ak.Array(np.zeros(0))],
    "inside-list": lambda: [
        ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 2, 2, 6])),
                option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0]),
            )
        ),
        ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 2, 2, 6])),
                option(
                    [4, -1, 2, 0, -1, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                ),
            )
        ),
    ],
    "inside-regular": lambda: [
        ak.Array(
            ak.contents.RegularArray(
                option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0]), 3
            )
        ),
        ak.Array(
            ak.contents.RegularArray(
                option(
                    [4, -1, 2, 0, -1, 6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                ),
                3,
            )
        ),
    ],
    "list-inside-option": lambda: [
        ak.Array(
            ak.contents.IndexedOptionArray(
                ak.index.Index64(np.array([0, -1, 1, 2, -1])),
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array([0, 2, 2, 5])),
                    ak.contents.NumpyArray(np.arange(5, dtype=np.float64)),
                ),
            )
        ),
        ak.Array(np.arange(5, dtype=np.float64)),
    ],
    "length-one-broadcast": lambda: [
        ak.Array(option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0])),
        ak.Array(np.array([100.0])),
    ],
    "parameters-on-the-option": lambda: [
        ak.Array(
            option([0, -1, 1, 2, -1, 3], [1.0, 2.0, 3.0, 4.0], {"__doc__": "left"})
        ),
        ak.Array(
            option([0, -1, 1, 2, -1, 3], [5.0, 6.0, 7.0, 8.0], {"__doc__": "left"})
        ),
    ],
}


@pytest.mark.parametrize("case", CASES)
def test_transform_projects_option_inputs_at_the_shared_valid_positions(case):
    arrays = CASES[case]()
    result = ak.transform(sum_leaves, *arrays)

    broadcast = ak.broadcast_arrays(*arrays)
    assert result.to_list() == expected_sum(*(x.to_list() for x in broadcast))
    assert result.layout.form.type == broadcast[0].layout.form.type
    assert (
        ak.transform(
            sum_leaves,
            *[ak.Array(x.layout.to_typetracer(forget_length=True)) for x in arrays],
        ).layout.form
        == result.layout.form
    )


@pytest.mark.parametrize("case", CASES)
def test_arithmetic_projects_option_inputs_at_the_shared_valid_positions(case):
    arrays = CASES[case]()
    result = arrays[0]
    for other in arrays[1:]:
        result = result + other

    broadcast = ak.broadcast_arrays(*arrays)
    assert result.to_list() == expected_sum(*(x.to_list() for x in broadcast))


def test_option_parameters_survive_the_projection():
    arrays = CASES["parameters-on-the-option"]()
    result = ak.transform(sum_leaves, *arrays)

    assert result.layout.parameters == {"__doc__": "left"}
