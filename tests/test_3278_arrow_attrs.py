# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


@pytest.fixture(
    params=[
        ak.Array(
            [[1.1, 2.2], [], [3.3]], attrs={"one": 1, "two": [2, {"three": None}]}
        ),
        ak.Array(
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": []}], attrs={"one": 1, "two": "2"}
        ),
    ],
    ids=["array", "record-array"],
)
def array(request):
    return request.param


def test_to_arrow_roundtrip(array):
    assert ak.from_arrow(ak.to_arrow(array)).attrs == array.attrs
    assert ak.from_arrow(ak.to_arrow_table(array)).attrs == array.attrs


def test_to_arrow_roundtrip_preserves_data(array):
    for through in ak.to_arrow, ak.to_arrow_table:
        result = ak.from_arrow(through(array))
        assert result.to_list() == array.to_list()
        assert result.type == array.type


def test_to_arrow_table_without_extensionarray(array):
    # the schema metadata is available whether or not there are extension arrays
    table = ak.to_arrow_table(array, extensionarray=False)
    assert ak.from_arrow(table).attrs == array.attrs


def test_to_arrow_without_extensionarray(array):
    # a plain Arrow array has nowhere to keep them, just as it has nowhere to
    # keep the array's type
    assert ak.from_arrow(ak.to_arrow(array, extensionarray=False)).attrs == {}


def test_no_attrs_leaves_metadata_alone(array):
    without = ak.Array(array.layout)
    assert ak.to_arrow(without).type == ak.to_arrow(without).type
    assert ak.from_arrow(ak.to_arrow(without)).attrs == {}

    schema = ak.to_arrow_table(without).schema
    assert schema.metadata is None or b"AWKWARD_ATTRS" not in schema.metadata
    assert ak.from_arrow(ak.to_arrow_table(without)).attrs == {}


def test_transient_attrs_are_dropped(array):
    class NotSerializable:
        pass

    array = ak.Array(
        array.layout, attrs={**array.attrs, "@transient": NotSerializable()}
    )

    assert ak.from_arrow(ak.to_arrow(array)).attrs == {
        k: v for k, v in array.attrs.items() if k != "@transient"
    }
    assert ak.from_arrow(ak.to_arrow_table(array)).attrs == {
        k: v for k, v in array.attrs.items() if k != "@transient"
    }


def test_only_transient_attrs():
    class NotSerializable:
        pass

    array = ak.Array([[1.1, 2.2], []], attrs={"@transient": NotSerializable()})

    assert ak.from_arrow(ak.to_arrow(array)).attrs == {}
    assert ak.from_arrow(ak.to_arrow_table(array)).attrs == {}


def test_explicit_attrs_take_precedence(array):
    result = ak.from_arrow(ak.to_arrow(array), attrs={"one": "overridden", "new": True})
    assert result.attrs == {**array.attrs, "one": "overridden", "new": True}


def test_from_arrow_lowlevel(array):
    # nothing to attach the attrs to; this must not raise
    assert isinstance(
        ak.from_arrow(ak.to_arrow(array), highlevel=False), ak.contents.Content
    )


def test_chunked_array_and_record_batch(array):
    chunked = pyarrow.chunked_array([ak.to_arrow(array), ak.to_arrow(array)])
    assert ak.from_arrow(chunked).attrs == array.attrs

    table = ak.to_arrow_table(array)
    assert ak.from_arrow(table.to_batches()[0]).attrs == array.attrs


def test_non_serializable_attrs_raise():
    array = ak.Array([[1.1, 2.2], []], attrs={"not_json": object()})

    with pytest.raises(TypeError):
        ak.to_arrow(array)
    with pytest.raises(TypeError):
        ak.to_arrow_table(array)


def test_arrow_table_attrs_are_readable_by_parquet(array, tmp_path):
    # ak.to_arrow_table stores attrs where ak.from_parquet looks for them
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    filename = os.path.join(tmp_path, "attrs.parquet")
    pyarrow_parquet.write_table(ak.to_arrow_table(array), filename)

    assert ak.from_parquet(filename).attrs == array.attrs
