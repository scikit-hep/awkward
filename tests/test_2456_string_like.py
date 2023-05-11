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
