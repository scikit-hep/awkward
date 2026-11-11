# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("reducer,which", [(ak.argmin, 0), (ak.argmax, 1)])
@pytest.mark.parametrize("axis", [0, -2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(
    "rows,expected,regular",
    [
        ([None, [1]], ([1], [1]), False),
        ([None, [1]], ([1], [1]), True),
        ([None, [3, 1], None, [1, 3], [1, 3]], ([3, 1], [1, 3]), False),
        ([None, [3, 1], None, [1, 3], [1, 3]], ([3, 1], [1, 3]), True),
        ([None, [], [3, 1], [1], None, [2, 4, 0]], ([3, 2, 5], [2, 5, 5]), False),
    ],
)
def test_missing_list_positions(
    reducer, which, axis, keepdims, nested, depth, rows, expected, regular
):
    expected = expected[which]
    empty = [None] * len(expected) if regular else []
    if depth == 2:
        rows = [None if row is None else [row] for row in rows]
        expected = [expected]
        empty = [empty] if regular else []
        if axis < 0:
            axis -= 1
    if nested:
        array = ak.Array([rows, rows, [None], rows])
        expected = [expected, expected, empty, expected]
        if axis >= 0:
            axis += 1
    else:
        array = ak.Array(rows)
    if regular:
        for inner_axis in range(-1, -depth - 1, -1):
            array = ak.to_regular(array, axis=inner_axis)
    if keepdims:
        expected = [[row] for row in expected] if nested else [expected]

    result = reducer(array, axis=axis, keepdims=keepdims)
    assert result.to_list() == expected
    assert (
        reducer(
            ak.Array(array.layout.to_typetracer(forget_length=True)),
            axis=axis,
            keepdims=keepdims,
        ).layout.form
        == result.layout.form
    )
