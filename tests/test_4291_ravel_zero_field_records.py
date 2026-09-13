# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

OPERATIONS = [
    pytest.param(ak.ravel, id="ravel"),
    pytest.param(lambda a: ak.flatten(a, axis=None), id="flatten-axis-None"),
]

LAYOUTS = [
    pytest.param(ak.Array([{}, {}, {}]).layout, id="records"),
    pytest.param(
        ak.contents.RecordArray([], fields=None, length=3),
        id="tuples",
    ),
    pytest.param(ak.Array([{}])[:0].layout, id="length-zero"),
    pytest.param(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3], dtype=np.int64)),
            ak.contents.RecordArray([], fields=[], length=3),
        ),
        id="var-list",
    ),
    pytest.param(
        ak.contents.RegularArray(ak.contents.RecordArray([], fields=[], length=6), 2),
        id="regular-list",
    ),
    pytest.param(ak.Array([{"a": {}, "b": {}}] * 3).layout, id="record-of-records"),
]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_zero_field_records_flatten_to_empty(operation, layout):
    result = operation(ak.Array(layout))
    assert result.to_list() == []
    assert str(result.type) == "0 * unknown"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_leaf_beside_zero_field_record_is_kept(operation):
    array = ak.Array([{"x": 1.1, "y": {}}, {"x": 2.2, "y": {}}])
    result = operation(array)
    assert result.to_list() == [1.1, 2.2]
    assert str(result.type) == "2 * float64"


def test_ravel_keeps_option_above_zero_field_record():
    result = ak.ravel(ak.Array([{}, None]))
    assert result.to_list() == [{}, None]
    assert str(result.type) == "2 * ?{}"


def test_flatten_axis_none_drops_option_above_zero_field_record():
    result = ak.flatten(ak.Array([{}, None]), axis=None)
    assert result.to_list() == []
    assert str(result.type) == "0 * unknown"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_typetracer_backend_is_preserved(operation):
    array = ak.Array(ak.Array([{}, {}]).layout.to_typetracer(forget_length=False))
    result = operation(array)
    assert result.layout.backend.name == "typetracer"
    assert str(result.type) == "0 * unknown"
