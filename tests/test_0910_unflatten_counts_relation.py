# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak


def test():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.array([3, 1, 0, 2])),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 6, 9, 12])),
            ak.contents.NumpyArray(np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])),
        ),
    )

    assert ak.operations.unflatten(
        layout,
        ak.operations.flatten(ak.operations.run_lengths(layout)),
        axis=1,
    ).to_list() == [[[3, 3, 3]], [[1, 1, 1]], [[0, 0, 0]], [[2, 2], [3]]]


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("order", [[0, 1, 2], [2, 1, 0], [0, 1, 0]])
@pytest.mark.parametrize("counts, split_row", [([2, 1, 1, 2], 1), ([2, 2, 1, 1], 2)])
def test_union(axis, order, counts, split_row):
    array = ak.concatenate([np.array([[1, 2]]), [["a", "b"]], np.array([[3, 4]])])[
        order
    ]
    assert array.layout.is_union

    expected = [[row] for row in ak.to_list(array)]
    expected[split_row] = [[value] for value in ak.to_list(array[split_row])]
    result = ak.unflatten(array, counts, axis=axis)
    assert ak.is_valid(result)
    assert result.to_list() == expected
