# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward.forms import NumpyForm, RecordForm
from awkward.types import NumpyType, RecordType

# Each record is (fields, primitives). Fields with the same name are matched in
# order, because array["a"] selects the first field named "a"; fields with
# different names are matched by name, in any order.
CASES = [
    ((["a", "a"], ["int64", "float64"]), (["a", "a"], ["int64", "float64"]), True),
    ((["a", "a"], ["int64", "int64"]), (["a", "a"], ["int64", "int64"]), True),
    ((["a", "a"], ["int64", "float64"]), (["a", "a"], ["float64", "int64"]), False),
    ((["a", "a"], ["int64", "int64"]), (["a"], ["int64"]), False),
    (
        (["a", "a", "b"], ["int64", "float64", "bool"]),
        (["b", "a", "a"], ["bool", "int64", "float64"]),
        True,
    ),
    (
        (["a", "a", "b"], ["int64", "float64", "bool"]),
        (["a", "b", "a"], ["int64", "bool", "float64"]),
        True,
    ),
    (
        (["a", "a", "b"], ["int64", "int64", "bool"]),
        (["a", "b", "b"], ["int64", "bool", "bool"]),
        False,
    ),
    ((["a", "b"], ["int64", "float64"]), (["b", "a"], ["float64", "int64"]), True),
    ((["a", "b"], ["int64", "float64"]), (["a", "b"], ["float64", "int64"]), False),
]


def record_type(fields, primitives):
    return RecordType([NumpyType(p) for p in primitives], fields)


def record_form(fields, primitives):
    return RecordForm([NumpyForm(p) for p in primitives], fields)


@pytest.mark.parametrize(("left", "right", "expected"), CASES)
def test_record_type_equality(left, right, expected):
    assert record_type(*left).is_equal_to(record_type(*right)) is expected
    assert record_type(*right).is_equal_to(record_type(*left)) is expected


@pytest.mark.parametrize(("left", "right", "expected"), CASES)
def test_record_form_equality(left, right, expected):
    assert (record_form(*left) == record_form(*right)) is expected
    assert (record_form(*right) == record_form(*left)) is expected


def test_layout_with_duplicate_fields():
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(np.array([1, 2])),
            ak.contents.NumpyArray(np.array([1.1, 2.2])),
        ],
        ["a", "a"],
    )
    assert layout.form.type.is_equal_to(layout.form.type) is True
    assert (layout.form == layout.form) is True
