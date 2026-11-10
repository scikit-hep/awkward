# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os
import runpy
import sys

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


def _ipc_roundtrip(array):
    sink = pyarrow.BufferOutputStream()
    schema = pyarrow.schema([("data", array.type)])
    with pyarrow.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(pyarrow.record_batch([array], schema=schema))
    return pyarrow.ipc.open_stream(sink.getvalue()).read_all().column("data")


def test_nested_attrs_are_snapshots():
    source = ak.Array([1, 2], attrs={"nested": {"values": [1]}, "tuple": (2, 3)})
    arrow = ak.to_arrow(source)
    expected = {"nested": {"values": [1]}, "tuple": [2, 3]}
    source.attrs["nested"]["values"].append(99)
    assert ak.from_arrow(arrow).attrs == expected
    assert ak.from_arrow(_ipc_roundtrip(arrow)).attrs == expected

    restored = ak.from_arrow(arrow)
    restored.attrs["nested"]["values"].append(42)
    assert ak.from_arrow(arrow).attrs == expected
    assert ak.from_arrow(_ipc_roundtrip(arrow)).attrs == expected
    arrow.type.attrs["nested"]["values"].append(17)
    assert ak.from_arrow(arrow).attrs == expected


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("empty_first", [False, True])
def test_batch_sequence_attrs(container, empty_first):
    array = ak.Array([1, 2, 3], attrs={"nested": {"value": [1]}, "source": "first"})
    table = ak.to_arrow_table(array)
    batches = table.to_batches(max_chunksize=1)
    if empty_first:
        batches.insert(0, batches[0].slice(0, 0))
    batches = container(batches)
    restored = ak.from_arrow(batches, attrs={"source": "override", "extra": True})
    assert restored.to_list() == array.to_list()
    assert restored.attrs == {
        "nested": {"value": [1]},
        "source": "override",
        "extra": True,
    }
    restored.attrs["nested"]["value"].append(99)
    assert ak.from_arrow(batches).attrs == array.attrs


def test_batch_sequence_first_batch_wins():
    first = ak.to_arrow_table(ak.Array([1], attrs={"source": "first"})).to_batches()[0]
    second = ak.to_arrow_table(ak.Array([2], attrs={"source": "second"})).to_batches()[
        0
    ]
    assert ak.from_arrow([first, second]).attrs == {"source": "first"}
    assert ak.from_arrow([]).attrs == {}


def test_metadata_helpers_without_pyarrow(monkeypatch):
    import awkward._connect.pyarrow

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    namespace = runpy.run_path(awkward._connect.pyarrow.__file__)
    for name in ("table_with_attrs", "attrs_from_schema_metadata"):
        with pytest.raises(NotImplementedError, match="requires pyarrow"):
            namespace[name]()


@pytest.mark.parametrize("iterative", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_parquet_shared_attrs_helpers(tmp_path, iterative, persistent):
    attrs = {"@transient": object()}
    if persistent:
        attrs["nested"] = {"values": [1]}
    first = ak.Array([1, 2], attrs=attrs)
    path = tmp_path / "attrs.parquet"
    if iterative:
        second = ak.Array([3], attrs={"nested": {"values": [99]}})
        ak.to_parquet_row_groups(iter([first, second]), path)
    else:
        ak.to_parquet(first, path)
    restored = ak.from_parquet(path)
    assert restored.to_list() == ([1, 2, 3] if iterative else [1, 2])
    assert restored.attrs == ({"nested": {"values": [1]}} if persistent else {})
