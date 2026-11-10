# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("missing", [0, 1, 2])
@pytest.mark.parametrize("k", [0, 1, 3, 5])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_regular_and_ragged_rows_have_original_indices(missing, k, mode):
    rows = [[1, 9], [5, 6]]
    rows.insert(missing, None)
    ragged = ak.Array(rows)
    regular = ak.to_regular(ragged, axis=1)
    expected_values, expected_indices = [], []
    for rank in range(min(k, len(rows))):
        values, indices = [], []
        for column in range(2):
            ordered = sorted(
                (i for i, row in enumerate(rows) if row is not None),
                key=lambda i: rows[i][column],
                reverse=mode == "largest",
            )
            index = ordered[rank] if rank < len(ordered) else None
            indices.append(index)
            values.append(None if index is None else rows[index][column])
        expected_values.append(values)
        expected_indices.append(indices)
    for array in (ragged, regular):
        result = ak.top_k(array, k, axis=0, mode=mode)
        assert result.values.to_list() == expected_values
        assert result.indices.to_list() == expected_indices
        assert ak.top_k_values(array, k, axis=0, mode=mode).to_list() == expected_values
        assert (
            ak.top_k_indices(array, k, axis=0, mode=mode).to_list() == expected_indices
        )
        traced = ak.top_k(
            array.layout.to_typetracer(forget_length=True), k, axis=0, mode=mode
        )
        assert traced.values.layout.form == result.values.layout.form
        assert traced.indices.layout.form == result.indices.layout.form


@pytest.mark.parametrize("axis", [1, -2])
def test_regular_missing_rows_below_outer_groups(axis):
    rows = [None, [1, 9], [5, 6]]
    ragged = ak.Array([rows, rows])
    regular = ak.to_regular(ragged, axis=2)
    for array in (ragged, regular):
        result = ak.top_k(array, 1, axis=axis)
        assert result.values.to_list() == [[[5, 9]], [[5, 9]]]
        assert result.indices.to_list() == [[[2, 1]], [[2, 1]]]
        traced = ak.top_k(array.layout.to_typetracer(forget_length=True), 1, axis=axis)
        assert traced.values.layout.form == result.values.layout.form
        assert traced.indices.layout.form == result.indices.layout.form
