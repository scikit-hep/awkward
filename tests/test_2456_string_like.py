# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_string_slice():
    fields = ak.Array(["x"])
    array = ak.Array([{"x": 1, "y": 2}])
    assert array[fields].to_list() == [{"x": 1}]


def test_stringlike_slice():
    behavior = {("__super__", "my-string"): "string"}
    fields = ak.with_parameter(["x"], "__array__", "my-string", behavior=behavior)
    array = ak.Array([{"x": 1, "y": 2}])
    assert array[fields].to_list() == [{"x": 1}]


@pytest.mark.skip(reason="test needs more work")
def test_stringlike_tolist():
    behavior = {("__super__", "my-string"): "string"}
    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-string", behavior=behavior
    )
    assert array.to_list() == ["bish", "bash", "bosh"]

    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-other-string", behavior=behavior
    )
    assert array.to_list() != ["bish", "bash", "bosh"]


def test_stringlike_flatten():
    behavior = {("__super__", "my-string"): "string"}

    # axis=None
    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-string", behavior=behavior
    )
    assert ak.flatten(array, axis=None).type.is_equal_to(
        ak.types.ArrayType(
            ak.types.ListType(
                ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                parameters={"__array__": "my-string"},
            ),
            3,
            None,
        )
    )

    # axis=1
    with pytest.raises(np.AxisError):
        ak.flatten(array, axis=1)


def test_stringlike_backend_array():
    behavior = {("__super__", "my-string"): "string"}
    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-string", behavior=behavior
    )
    assert ak.to_numpy(array).tolist() == ["bish", "bash", "bosh"]


def test_stringlike_values_astype():
    behavior = {("__super__", "my-string"): "string"}
    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-string", behavior=behavior
    )
    assert ak.values_astype(array, np.int64).tolist() == ["bish", "bash", "bosh"]

    array = ak.with_parameter(["bish", "bash", "bosh"], "__array__", "my-other-string")
    assert ak.values_astype(array, np.float32).tolist() == ["bish", "bash", "bosh"]


def test_stringlike_to_arrow_table():
    behavior = {("__super__", "my-string"): "string"}
    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-string", behavior=behavior
    )
    table = ak.to_arrow_table(array, extensionarray=False)

    # Fails
    assert table.schema.to_string() == ": large_string not null"
    array = ak.with_parameter(
        ["bish", "bash", "bosh"], "__array__", "my-other-string", behavior=behavior
    )
    table = ak.to_arrow_table(array, extensionarray=False)
    assert (
        table.schema.to_string()
        == ": large_list<item: uint8 not null> not null\n  child 0, item: uint8 not null"
    )


def test_string_broadcasting():
    result = ak.broadcast_arrays(["he", "lo"], ["w", "orld"])
    assert result[0].tolist() == ["he", "lo"]
    assert result[1].tolist() == ["w", "orld"]

    with pytest.raises(ValueError):
        ak.broadcast_arrays(["he", "lo"], [1, 2, 3])

    result = ak.broadcast_arrays(["he", "lo"], [[1, 2, 3], [4]])
    assert result[0].tolist() == [["he", "he", "he"], ["lo"]]
    assert result[1].tolist() == [[1, 2, 3], [4]]


def test_stringlike_broadcasting():
    behavior = {("__super__", "my-string"): "string"}
    result = ak.broadcast_arrays(
        ak.with_parameter(["he", "lo"], "__array__", "my-string"),
        ak.with_parameter(["w", "orld"], "__array__", "my-string"),
        behavior=behavior,
    )
    assert result[0].tolist() == ["he", "lo"]
    assert result[1].tolist() == ["w", "orld"]

    with pytest.raises(ValueError):
        ak.broadcast_arrays(
            ak.with_parameter(["he", "lo"], "__array__", "my-string"),
            [1, 2, 3],
            behavior=behavior,
        )

    result = ak.broadcast_arrays(
        ak.with_parameter(["he", "lo"], "__array__", "my-string"),
        [[1, 2, 3], [4]],
        behavior=behavior,
    )
    assert result[0].tolist() == [["he", "he", "he"], ["lo"]]
    assert result[1].tolist() == [[1, 2, 3], [4]]
