# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


def empty_content(kind):
    if kind == "list":
        return ak.Array([[1, 2]])[:0].layout
    if kind == "string":
        return ak.Array(["hello"])[:0].layout
    if kind == "bytes":
        return ak.Array([b"hello"])[:0].layout
    if kind == "number":
        return ak.contents.NumpyArray(np.array([], dtype=np.int64))
    if kind == "regular":
        return ak.contents.RegularArray(empty_content("number"), 2)
    if kind == "record":
        return ak.Array([{"x": [1, 2]}])[:0].layout
    if kind == "unknown":
        return ak.contents.EmptyArray()
    raise AssertionError(kind)


def validate_full(array):
    # pyarrow < 18 aborts on full validation of string-backed extension arrays.
    full = parse_version(pyarrow.__version__) >= parse_version("18") or not isinstance(
        array.type, pyarrow.ExtensionType
    )
    array.validate(full=full)


def all_null(kind, length, dtype=np.int64, parameters=None):
    return ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index(np.full(length, -1, dtype=dtype)),
            empty_content(kind),
            parameters=parameters,
        )
    )


@pytest.mark.parametrize(
    "kind", ["list", "string", "bytes", "number", "regular", "record", "unknown"]
)
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_to_arrow_empty_content(kind, length, extensionarray):
    array = all_null(kind, length)
    assert ak.validity_error(array) == ""
    reference = ak.to_arrow(array[:0], extensionarray=extensionarray)
    out = ak.to_arrow(array, extensionarray=extensionarray)
    assert len(out) == length
    assert out.null_count == length
    assert out.type == reference.type
    assert out.to_pylist() == [None] * length
    returned = ak.from_arrow(out)
    assert ak.to_list(returned) == [None] * length
    # Without extension metadata, an empty Arrow array cannot distinguish
    # optional from non-optional content when no values carry a null bit.
    if extensionarray or length != 0:
        assert str(ak.type(returned)) == str(ak.type(array))


@pytest.mark.parametrize("kind", ["list", "string", "bytes"])
@pytest.mark.parametrize("extensionarray", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_conversion_options(kind, extensionarray, dtype):
    array = all_null(kind, 3, dtype)
    options = {
        "list_to32": True,
        "string_to32": True,
        "bytestring_to32": True,
        "count_nulls": False,
    }
    out = ak.to_arrow(array, extensionarray=extensionarray, **options)
    reference = ak.to_arrow(array[:0], extensionarray=extensionarray, **options)
    assert out.type == reference.type
    assert len(out) == out.null_count == 3
    assert out.to_pylist() == [None, None, None]
    assert ak.to_list(ak.from_arrow(out)) == [None, None, None]


@pytest.mark.parametrize("kind", ["list", "string", "bytes"])
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("extensionarray", [False, True])
def test_parquet_empty_content(tmp_path, kind, length, extensionarray):
    array = all_null(kind, length)
    path = tmp_path / "nulls.parquet"
    ak.to_parquet(array, path, extensionarray=extensionarray)
    returned = ak.from_parquet(path)
    assert ak.to_list(returned) == [None] * length
    assert str(ak.type(returned)) == str(ak.type(array))


@pytest.mark.parametrize("kind", ["list", "string", "bytes", "number"])
@pytest.mark.parametrize("length", [0, 3])
def test_extension_metadata(tmp_path, kind, length):
    content = empty_content(kind).copy(
        parameters={**empty_content(kind).parameters, "content-tag": "child"}
    )
    array = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.full(length, -1, dtype=np.int64)),
            content,
            parameters={"mask-tag": "parent"},
        ),
        attrs={"sample": {"id": 42}},
    )
    out = ak.to_arrow(array)
    assert out.type.mask_type == "IndexedOptionArray"
    assert out.type.mask_parameters == {"mask-tag": "parent"}
    assert out.type.node_parameters == content.parameters
    assert out.type.attrs == array.attrs
    returned = ak.from_arrow(out)
    assert returned.attrs == array.attrs
    assert str(ak.type(returned)) == str(ak.type(array))
    path = tmp_path / "metadata.parquet"
    ak.to_parquet(array, path)
    returned = ak.from_parquet(path)
    assert returned.attrs == array.attrs
    assert str(ak.type(returned)) == str(ak.type(array))


def test_record_metadata():
    content = ak.with_name(ak.Array([{"x": [1, 2]}]), "Particle")[:0].layout
    array = ak.Array(
        ak.contents.IndexedOptionArray(ak.index.Index64(np.array([-1, -1])), content)
    )
    out = ak.to_arrow(array)
    assert out.type.node_parameters == {"__record__": "Particle"}
    assert out.null_count == len(out) == 2
    assert ak.to_list(ak.from_arrow(out)) == [None, None]
    assert str(ak.type(ak.from_arrow(out))) == str(ak.type(array))


@pytest.mark.parametrize("extensionarray", [False, True])
def test_public_concatenate_reproducer(tmp_path, extensionarray):
    array = ak.concatenate([ak.Array([None]), ak.Array([[1]])[:0]])
    out = ak.to_arrow(array, extensionarray=extensionarray)
    assert len(out) == out.null_count == 1
    assert out.to_pylist() == [None]
    path = tmp_path / "concatenated.parquet"
    ak.to_parquet(array, path, extensionarray=extensionarray)
    assert ak.to_list(ak.from_parquet(path)) == [None]


@pytest.mark.parametrize("extensionarray", [False, True])
@pytest.mark.parametrize("example", [[[1, 2]], ["hello"], [b"hello"]])
@pytest.mark.parametrize("index", [[-1, -1], [0, -1]])
def test_nonempty_content_controls(extensionarray, example, index):
    array = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array(index)), ak.Array(example).layout
        )
    )
    out = ak.to_arrow(array, extensionarray=extensionarray)
    assert out.to_pylist() == ak.to_list(array)
    assert ak.to_list(ak.from_arrow(out)) == ak.to_list(array)


@pytest.mark.parametrize("kind", ["list", "string", "bytes", "number"])
def test_invalid_indexed_array_does_not_invent_nulls(kind):
    array = ak.Array(
        ak.contents.IndexedArray(ak.index.Index64(np.array([0])), empty_content(kind))
    )
    assert ak.validity_error(array) != ""
    with pytest.raises((IndexError, ValueError)):
        ak.to_arrow(array)


@pytest.mark.parametrize("kind", ["list", "string", "bytes", "number"])
@pytest.mark.parametrize("length", [0, 3])
@pytest.mark.parametrize("extensionarray", [False, True])
@pytest.mark.parametrize("categorical_as_dictionary", [False, True])
def test_empty_categorical_content(
    tmp_path, kind, length, extensionarray, categorical_as_dictionary
):
    array = all_null(kind, length, parameters={"__array__": "categorical"})
    options = {
        "extensionarray": extensionarray,
        "categorical_as_dictionary": categorical_as_dictionary,
    }
    out = ak.to_arrow(array, **options)
    validate_full(out)
    assert len(out) == out.null_count == length
    assert out.to_pylist() == [None] * length
    reference = ak.to_arrow(all_null(kind, 0), extensionarray=extensionarray)
    if extensionarray:
        assert out.type.mask_parameters == {"__array__": "categorical"}
        assert out.type.node_parameters == empty_content(kind).parameters
        assert out.type.storage_type == reference.type.storage_type
    else:
        assert out.type == reference.type
        assert ak.to_list(ak.from_arrow(out)) == [None] * length
    path = tmp_path / "categorical.parquet"
    ak.to_parquet(array, path, **options)
    table = pyarrow.parquet.read_table(path)
    assert table.num_rows == length
    assert table.column(0).to_pylist() == [None] * length
    # Projected categorical extension readback has a separate, existing
    # mask-parameter limitation, also present with nonempty content.
    if not extensionarray:
        assert ak.to_list(ak.from_parquet(path)) == [None] * length


def nonempty_categorical(kind):
    if kind == "list":
        content = ak.Array([[1, 2]]).layout
    elif kind == "string":
        content = ak.Array(["hello"]).layout
    elif kind == "bytes":
        content = ak.Array([b"hello"]).layout
    else:
        content = ak.Array([42]).layout
    return ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([-1, 0])),
            content,
            parameters={"__array__": "categorical"},
        )
    )


@pytest.mark.parametrize("kind", ["list", "string", "bytes", "number"])
@pytest.mark.parametrize("extensionarray", [False, True])
@pytest.mark.parametrize("categorical_as_dictionary", [False, True])
def test_nonempty_dictionary_controls(kind, extensionarray, categorical_as_dictionary):
    array = nonempty_categorical(kind)
    out = ak.to_arrow(
        array,
        extensionarray=extensionarray,
        categorical_as_dictionary=categorical_as_dictionary,
    )
    storage_type = out.type.storage_type if extensionarray else out.type
    assert pyarrow.types.is_dictionary(storage_type) == categorical_as_dictionary
    validate_full(out)
    assert out.to_pylist() == ak.to_list(array)
    if not extensionarray or categorical_as_dictionary:
        assert ak.to_list(ak.from_arrow(out)) == ak.to_list(array)


@pytest.mark.parametrize("kind", ["string", "bytes", "number"])
@pytest.mark.parametrize("categorical_as_dictionary", [False, True])
def test_nonempty_dictionary_parquet_controls(
    tmp_path, kind, categorical_as_dictionary
):
    array = nonempty_categorical(kind)
    path = tmp_path / "nonempty-dictionary.parquet"
    ak.to_parquet(
        array,
        path,
        extensionarray=False,
        categorical_as_dictionary=categorical_as_dictionary,
    )
    assert ak.to_list(ak.from_parquet(path)) == ak.to_list(array)
