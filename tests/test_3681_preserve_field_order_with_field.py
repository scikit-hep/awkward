# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_simple_field_replacement_preserves_order():
    """Test that replacing an existing field preserves its position in the field order."""
    array = ak.Array([{"x": 1, "y": 2, "z": 3}])

    # Replace middle field
    result = ak.with_field(array, [100], "y")
    assert ak.fields(result) == ["x", "y", "z"]
    assert result.to_list() == [{"x": 1, "y": 100, "z": 3}]

    # Replace first field
    result = ak.with_field(array, [100], "x")
    assert ak.fields(result) == ["x", "y", "z"]
    assert result.to_list() == [{"x": 100, "y": 2, "z": 3}]

    # Replace last field
    result = ak.with_field(array, [100], "z")
    assert ak.fields(result) == ["x", "y", "z"]
    assert result.to_list() == [{"x": 1, "y": 2, "z": 100}]


def test_simple_field_addition_appends():
    """Test that adding a new field appends it to the end."""
    array = ak.Array([{"x": 1, "y": 2, "z": 3}])

    result = ak.with_field(array, [100], "w")
    assert ak.fields(result) == ["x", "y", "z", "w"]
    assert result.to_list() == [{"x": 1, "y": 2, "z": 3, "w": 100}]


def test_setitem_replacement_preserves_order():
    """Test that using array['field'] = value preserves field order."""
    array = ak.Array([{"x": 1, "y": 2, "z": 3}])

    # Replace middle field using setitem
    array["y"] = [100]
    assert ak.fields(array) == ["x", "y", "z"]
    assert array.to_list() == [{"x": 1, "y": 100, "z": 3}]


def test_setitem_addition_appends():
    """Test that using array['new_field'] = value appends the field."""
    array = ak.Array([{"x": 1, "y": 2, "z": 3}])

    array["w"] = [100]
    assert ak.fields(array) == ["x", "y", "z", "w"]
    assert array.to_list() == [{"x": 1, "y": 2, "z": 3, "w": 100}]


def test_nested_field_replacement_preserves_order():
    """Test that replacing a nested field preserves order at both levels."""
    array = ak.Array([{"a": {"x": 1, "y": 2, "z": 3}, "b": 10, "c": 20}])

    # Replace nested field "y" inside "a"
    result = ak.with_field(array, [100], ["a", "y"])
    assert ak.fields(result) == ["a", "b", "c"]
    assert ak.fields(result["a"]) == ["x", "y", "z"]
    assert result.to_list() == [{"a": {"x": 1, "y": 100, "z": 3}, "b": 10, "c": 20}]


def test_nested_field_replacement_with_setitem():
    """Test nested field replacement using setitem syntax."""
    array = ak.Array([{"a": {"x": 1, "y": 2, "z": 3}, "b": 10, "c": 20}])

    # Replace nested field using setitem
    array["a", "y"] = [100]
    assert ak.fields(array) == ["a", "b", "c"]
    assert ak.fields(array["a"]) == ["x", "y", "z"]
    assert array.to_list() == [{"a": {"x": 1, "y": 100, "z": 3}, "b": 10, "c": 20}]


def test_nested_field_addition():
    """Test that adding a new nested field appends it."""
    array = ak.Array([{"a": {"x": 1, "y": 2, "z": 3}, "b": 10, "c": 20}])

    # Add new nested field
    result = ak.with_field(array, [100], ["a", "w"])
    assert ak.fields(result) == ["a", "b", "c"]
    assert ak.fields(result["a"]) == ["x", "y", "z", "w"]
    assert result.to_list() == [
        {"a": {"x": 1, "y": 2, "z": 3, "w": 100}, "b": 10, "c": 20}
    ]


def test_multiple_replacements_preserve_order():
    """Test that multiple sequential replacements preserve order."""
    array = ak.Array([{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}])

    # Replace multiple fields in different orders
    result = array
    result = ak.with_field(result, [20], "b")
    result = ak.with_field(result, [40], "d")
    result = ak.with_field(result, [10], "a")

    assert ak.fields(result) == ["a", "b", "c", "d", "e"]
    assert result.to_list() == [{"a": 10, "b": 20, "c": 3, "d": 40, "e": 5}]


def test_replacement_with_array_data():
    """Test field replacement with various array structures."""
    array = ak.Array(
        [
            {"x": 1, "y": [1, 2], "z": 3},
            {"x": 2, "y": [3, 4, 5], "z": 4},
        ]
    )

    # Replace field with different array structure
    result = ak.with_field(array, [[10, 20], [30, 40, 50]], "y")
    assert ak.fields(result) == ["x", "y", "z"]
    assert result.to_list() == [
        {"x": 1, "y": [10, 20], "z": 3},
        {"x": 2, "y": [30, 40, 50], "z": 4},
    ]


def test_replacement_with_scalar_broadcast():
    """Test field replacement with scalar broadcasting."""
    array = ak.Array(
        [
            {"x": 1, "y": 2, "z": 3},
            {"x": 4, "y": 5, "z": 6},
            {"x": 7, "y": 8, "z": 9},
        ]
    )

    # Replace with scalar (should broadcast)
    result = ak.with_field(array, 100, "y")
    assert ak.fields(result) == ["x", "y", "z"]
    assert result.to_list() == [
        {"x": 1, "y": 100, "z": 3},
        {"x": 4, "y": 100, "z": 6},
        {"x": 7, "y": 100, "z": 9},
    ]


def test_tuple_field_replacement():
    """Test replacing a field in a tuple (which converts it to a record)."""
    array = ak.Array([(1, 2, 3)])

    # Replacing field "1" should preserve position
    result = ak.with_field(array, [100], "1")
    assert ak.fields(result) == ["0", "1", "2"]
    assert result.to_list() == [{"0": 1, "1": 100, "2": 3}]


def test_single_field_replacement():
    """Test replacing the only field in a record."""
    array = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}])

    result = ak.with_field(array, [100, 200, 300], "x")
    assert ak.fields(result) == ["x"]
    assert result.to_list() == [{"x": 100}, {"x": 200}, {"x": 300}]


def test_deeply_nested_replacement():
    """Test replacing a deeply nested field."""
    array = ak.Array([{"a": {"b": {"x": 1, "y": 2, "z": 3}, "c": 10}, "d": 100}])

    # Replace deeply nested field
    result = ak.with_field(array, [999], ["a", "b", "y"])
    assert ak.fields(result) == ["a", "d"]
    assert ak.fields(result["a"]) == ["b", "c"]
    assert ak.fields(result["a", "b"]) == ["x", "y", "z"]
    assert result.to_list() == [
        {"a": {"b": {"x": 1, "y": 999, "z": 3}, "c": 10}, "d": 100}
    ]


def test_where_none_always_appends():
    """Test that where=None always appends a new field."""
    array = ak.Array([{"x": 1, "y": 2}])

    # where=None should always append
    result = ak.with_field(array, [100], where=None)
    assert ak.fields(result) == ["x", "y", "2"]
    assert result.to_list() == [{"x": 1, "y": 2, "2": 100}]

    # Multiple where=None should keep appending
    result = ak.with_field(result, [200], where=None)
    assert ak.fields(result) == ["x", "y", "2", "3"]
    assert result.to_list() == [{"x": 1, "y": 2, "2": 100, "3": 200}]


def test_mixed_operations():
    """Test a mix of additions and replacements."""
    array = ak.Array([{"x": 1, "y": 2, "z": 3}])

    # Add a new field
    result = ak.with_field(array, [10], "a")
    assert ak.fields(result) == ["x", "y", "z", "a"]

    # Replace an existing field
    result = ak.with_field(result, [20], "y")
    assert ak.fields(result) == ["x", "y", "z", "a"]

    # Add another new field
    result = ak.with_field(result, [30], "b")
    assert ak.fields(result) == ["x", "y", "z", "a", "b"]

    # Replace the first field
    result = ak.with_field(result, [100], "x")
    assert ak.fields(result) == ["x", "y", "z", "a", "b"]

    assert result.to_list() == [{"x": 100, "y": 20, "z": 3, "a": 10, "b": 30}]


def test_replacement_in_list_of_records():
    """Test field replacement in a list of records."""
    array = ak.Array(
        [
            [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}],
            [{"x": 7, "y": 8, "z": 9}],
        ]
    )

    result = ak.with_field(array, [[100, 200], [300]], "y")
    assert ak.fields(result) == ["x", "y", "z"]
    assert result.to_list() == [
        [{"x": 1, "y": 100, "z": 3}, {"x": 4, "y": 200, "z": 6}],
        [{"x": 7, "y": 300, "z": 9}],
    ]
