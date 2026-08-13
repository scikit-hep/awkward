# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

OPERATIONS = [
    pytest.param(ak.ravel, id="ravel"),
    pytest.param(lambda a: ak.flatten(a, axis=None), id="flatten-axis-None"),
]


def _union_of_float_and_zero_field_record(tags, index):
    return ak.contents.UnionArray(
        ak.index.Index8(np.array(tags, dtype=np.int8)),
        ak.index.Index64(np.array(index, dtype=np.int64)),
        [
            ak.contents.NumpyArray(np.array([1.5, 2.5])),
            ak.contents.RecordArray([], fields=[], length=2),
        ],
    )


CASES = [
    pytest.param(
        ak.Array([1, "s"])[:0],
        [],
        "0 * unknown",
        id="empty-union-with-string-branch",
    ),
    pytest.param(
        ak.Array([1, {"a": 1, "b": 2}])[:0],
        [],
        "0 * unknown",
        id="empty-union-with-record-branch",
    ),
    pytest.param(
        ak.Array([[1, "s"], []])[1:],
        [],
        "0 * unknown",
        id="union-below-list-reached-by-no-values",
    ),
    pytest.param(
        ak.Array([1, [2, 3]])[:0],
        [],
        "0 * int64",
        id="empty-union-with-numeric-branches",
    ),
    pytest.param(
        ak.Array(_union_of_float_and_zero_field_record(tags=[1, 1], index=[0, 1])),
        [],
        "0 * unknown",
        id="union-where-only-zero-field-records-are-reached",
    ),
    pytest.param(
        ak.Array(_union_of_float_and_zero_field_record(tags=[0, 1], index=[0, 0])),
        [1.5],
        "1 * float64",
        id="union-zero-field-record-beside-leaf",
    ),
]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(("array", "expected", "expected_type"), CASES)
def test_union_flattens_to_expected(operation, array, expected, expected_type):
    result = operation(array)
    assert result.to_list() == expected
    assert str(result.type) == expected_type
