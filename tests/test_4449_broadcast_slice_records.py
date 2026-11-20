# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("tuple_record", [False, True])
def test_record_list_fields(axis, tuple_record):
    fields = [ak.to_layout([[1, 2], [3]]), ak.to_layout([[10, 20], [30]])]
    layout = ak.contents.RecordArray(fields, None if tuple_record else ["a", "b"])
    array = ak.Array(layout)
    indices = ak.Array([[[1, 0]], [[0]]])
    result = ak.broadcast_slice(array, indices, axis)
    expected = (
        [([[2, 1]], [[20, 10]]), ([[3]], [[30]])]
        if tuple_record
        else [{"a": [[2, 1]], "b": [[20, 10]]}, {"a": [[3]], "b": [[30]]}]
    )
    assert result.to_list() == expected
    tracer = ak.broadcast_slice(
        layout.to_typetracer(forget_length=True),
        indices.layout.to_typetracer(forget_length=True),
        axis,
        highlevel=False,
    )
    assert result.layout.form == tracer.form


def test_nested_record_fields_and_metadata():
    array = ak.zip(
        {
            "a": ak.zip({"x": [[1, 2], [3]]}, depth_limit=1),
            "b": [[10, 20], [30]],
        },
        depth_limit=1,
        with_name="Collections",
    )
    result = ak.broadcast_slice(array, [[[1, 0]], [[0]]], attrs={"source": "test"})
    assert result.to_list() == [
        {"a": {"x": [[2, 1]]}, "b": [[20, 10]]},
        {"a": {"x": [[3]]}, "b": [[30]]},
    ]
    assert ak.parameters(result)["__record__"] == "Collections"
    assert result.attrs["source"] == "test"


def test_record_fields_at_deeper_axis():
    array = ak.Array([[{"a": [1, 2], "b": [10, 20]}], [{"a": [3], "b": [30]}]])
    indices = ak.Array([[[[1, 0]]], [[[0]]]])
    result = ak.broadcast_slice(array, indices, axis=2)
    assert result.to_list() == [
        [{"a": [[2, 1]], "b": [[20, 10]]}],
        [{"a": [[3]], "b": [[30]]}],
    ]
    traced = ak.broadcast_slice(
        array.layout.to_typetracer(forget_length=True),
        indices.layout.to_typetracer(forget_length=True),
        axis=2,
    )
    assert traced.layout.form == result.layout.form


def test_missing_field_collections_and_indices():
    array = ak.zip({"a": [[1, 2], None], "b": [[10, 20], [30]]}, depth_limit=1)
    result = ak.broadcast_slice(array, [[[None, 0]], [[0]]])
    assert result.to_list() == [
        {"a": [[None, 1]], "b": [[None, 10]]},
        {"a": None, "b": [[30]]},
    ]


def test_record_field_bounds_are_checked_independently():
    array = ak.zip({"a": [[1, 2]], "b": [[10]]}, depth_limit=1)
    with pytest.raises(IndexError):
        ak.broadcast_slice(array, [[[1]]])
