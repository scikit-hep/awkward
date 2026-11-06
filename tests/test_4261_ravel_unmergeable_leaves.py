# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import datetime
import re

import numpy as np
import pytest

import awkward as ak

OPERATIONS = [
    pytest.param(ak.ravel, id="ravel"),
    pytest.param(lambda a: ak.flatten(a, axis=None), id="flatten-axis-None"),
]


def _record(a, b):
    return ak.Array({"x": np.array([1], dtype=a), "y": np.array([1], dtype=b)})


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "data",
    [
        [{"x": 1, "y": "s"}],
        [{"x": "s", "y": 1}],
        [1, "s", 2.5],
        ["s", 1, 2.5],
    ],
)
def test_int_and_string(operation, data):
    with pytest.raises(ValueError, match="into a single flat array"):
        operation(ak.Array(data))


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("float64", "timedelta64[ms]"),
        ("datetime64[s]", "int64"),
        ("timedelta64[ms]", "timedelta64[M]"),
        ("datetime64[s]", "timedelta64[s]"),
        ("datetime64[s]", "datetime64[ms]"),
        ("timedelta64[s]", "timedelta64[ms]"),
        ("int64", "timedelta64[ms]"),
    ],
)
def test_unmergeable_dtypes(operation, a, b):
    with pytest.raises(
        ValueError, match=re.escape(f"cannot merge types {a} and {b} into")
    ):
        operation(_record(a, b))
    with pytest.raises(
        ValueError, match=re.escape(f"cannot merge types {b} and {a} into")
    ):
        operation(_record(b, a))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_bytes_and_string(operation):
    with pytest.raises(ValueError, match="cannot merge types bytes and string into"):
        operation(ak.Array([{"x": b"b", "y": "s"}]))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_different_depths_int_and_string(operation):
    with pytest.raises(ValueError, match="into a single flat array"):
        operation(ak.Array([[[1]], ["s"]]))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_different_depths_int(operation):
    result = operation(ak.Array([[1], [[2]]]))
    assert result.to_list() == [1, 2]
    assert str(result.type) == "2 * int64"


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "array",
    [
        ak.Array({"x": [1], "y": [[]], "z": ["s"]}),
        ak.Array({"y": [[]], "x": [1], "z": ["s"]}),
        ak.Array({"y": [None], "x": [1], "z": ["s"]}),
        ak.Array(
            ak.contents.RecordArray(
                [
                    ak.contents.ByteMaskedArray(
                        ak.index.Index8(np.array([], dtype=np.int8)),
                        ak.contents.EmptyArray(),
                        valid_when=True,
                    ),
                    ak.contents.NumpyArray(np.array([], dtype=np.int64)),
                    ak.Array(["s"])[:0].layout,
                ],
                ["y", "x", "z"],
                length=0,
            )
        ),
    ],
)
def test_empty_leaf_beside_unmergeable(operation, array):
    with pytest.raises(ValueError, match="cannot merge types int64 and string into"):
        operation(array)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_empty_leaf_beside_mergeable(operation):
    result = operation(ak.Array({"x": [1], "y": [[]], "z": [2.5]}))
    assert result.to_list() == [1.0, 2.5]
    assert str(result.type) == "2 * float64"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_int_and_float(operation):
    result = operation(_record("int64", "float64"))
    assert result.to_list() == [1.0, 1.0]
    assert str(result.type) == "2 * float64"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_same_datetime_unit(operation):
    result = operation(_record("datetime64[s]", "datetime64[s]"))
    assert result.to_list() == [datetime.datetime(1970, 1, 1, 0, 0, 1)] * 2
    assert str(result.type) == "2 * datetime64[s]"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_string_and_string(operation):
    result = operation(ak.Array([{"x": "a", "y": "b"}]))
    assert result.to_list() == ["a", "b"]
    assert str(result.type) == "2 * string"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_typetracer_int_and_string(operation):
    array = ak.Array([{"x": 1, "y": "s"}])
    with pytest.raises(ValueError, match="cannot merge types int64 and string into"):
        operation(ak.Array(array.layout.to_typetracer(forget_length=True)))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_typetracer_int_and_float(operation):
    array = _record("int64", "float64")
    result = operation(ak.Array(array.layout.to_typetracer(forget_length=True)))
    assert result.layout.backend.name == "typetracer"
    assert result.layout.form == operation(array).layout.form
