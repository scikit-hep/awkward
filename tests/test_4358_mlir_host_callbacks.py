# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Execute actual device callback bodies on the host for contract/line coverage.

These tests do not validate MLIR compilation or GPU execution; tests-cuda does.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend


@pytest.fixture(scope="module")
def callbacks():
    path = Path(ak.__file__).resolve().parent / "_connect/cuda/_compute.py"
    names = {
        "make_segment_views",
        "normalize_index_dtype",
        "infer_complex_dtype",
        "awkward_reduce_argmin",
        "awkward_reduce_argmax",
        "awkward_reduce_argmin_complex",
        "awkward_reduce_argmax_complex",
        "lexargmin_complex",
        "lexargmax_complex",
        "awkward_BitMaskedArray_to_IndexedOptionArray",
        "awkward_BitMaskedArray_to_ByteMaskedArray",
    }
    body = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            # Execute the unchanged function body, without a device compiler.
            node.decorator_list = []
            body.append(node)
    assert {node.name for node in body} == names

    def transform(*, d_in, d_out, op, num_items):
        for i in range(num_items):
            d_out[i] = op(i)

    namespace = {
        "__name__": "awkward._connect.cuda._compute",
        "np": np,
        "cp": np,
        "CountingIterator": lambda start: start,
        "unary_transform": transform,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def bitmask_specification():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "kernel-specification.yml"
    names = {
        "awkward_BitMaskedArray_to_IndexedOptionArray",
        "awkward_BitMaskedArray_to_ByteMaskedArray",
    }
    namespace = {"uint8": np.uint8}
    for kernel in yaml.safe_load(path.read_text())["kernels"]:
        if kernel["name"] in names:
            exec(compile(kernel["definition"], str(path), "exec"), namespace)
    assert names <= namespace.keys()
    return namespace


@pytest.mark.parametrize("operation", ["argmin", "argmax"])
@pytest.mark.parametrize("origin", [0, 3])
@pytest.mark.parametrize("offset_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_numeric_callbacks_match_cpu(callbacks, operation, origin, offset_dtype, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        small = np.array([np.nan, 1, 2, 1, np.nan, -1, -np.inf, np.inf, 0], dtype=dtype)
        low, high = -3, 3
    else:
        low, high = (
            (False, True)
            if dtype.kind == "b"
            else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        )
        small = np.array([high, low, high, low, high, low, high, 0, 0], dtype=dtype)
    large = np.zeros(513, dtype=dtype)
    large[255], large[512] = high, low
    values = np.concatenate([np.zeros(origin, dtype=dtype), small, large])
    offsets = np.array(
        [origin, origin, origin + 3, origin + 7, origin + 9, len(values)],
        dtype=offset_dtype,
    )
    actual = np.full(7, 123, dtype=np.int64)
    expected = actual.copy()
    # Native kernels use int8 for bool and float32 for float16, preserving all
    # input values exactly while exercising the callback with its original dtype.
    native = (
        values.astype(np.int8 if dtype.kind == "b" else np.float32)
        if dtype in (np.dtype(bool), np.dtype(np.float16))
        else values
    )
    key = (
        f"awkward_reduce_{operation}",
        np.int64,
        native.dtype.type,
        np.int64,
        np.int64,
    )
    error = NumpyBackend.instance()[key](
        expected, native, offsets.astype(np.int64), offsets[:-1].astype(np.int64), 5
    )
    assert not error.str
    for _ in range(2):
        callbacks[f"awkward_reduce_{operation}"](
            actual, values, offsets, offsets[:-1], 5
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation", ["argmin", "argmax"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("origin", [0, 3])
def test_complex_callbacks_match_cpu(callbacks, operation, dtype, origin):
    large = np.full(513, 2 + 3j, dtype=dtype)
    large[256], large[512] = 2 + 4j, -1 + 100j
    values = np.concatenate(
        [
            np.zeros(origin, dtype=dtype),
            np.array(
                [1 + 2j, 1 + 3j, 1 + 3j, -2 + 5j, -2 + 5j, 3 - 1j, 3 - 1j, 7],
                dtype=dtype,
            ),
            large,
        ]
    )
    offsets = np.array(
        [origin, origin, origin + 3, origin + 7, origin + 8, len(values)],
        dtype=np.int64,
    )
    real = values.view(np.float32 if dtype is np.complex64 else np.float64)
    actual = np.full(7, 123, dtype=np.int64)
    expected = actual.copy()
    key = (f"awkward_reduce_{operation}_complex", np.int64, real.dtype.type, np.int64)
    error = NumpyBackend.instance()[key](expected, real, offsets, 5)
    assert not error.str
    for _ in range(2):
        callbacks[f"awkward_reduce_{operation}_complex"](actual, real, offsets, 5)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("validwhen", [False, True])
@pytest.mark.parametrize("lsb_order", [False, True])
@pytest.mark.parametrize("length", [0, 1, 65, 256])
@pytest.mark.parametrize(
    "kind,dtype", [("ByteMaskedArray", np.int8), ("IndexedOptionArray", np.int64)]
)
def test_bitmask_callbacks_match_spec(
    callbacks, bitmask_specification, validwhen, lsb_order, length, kind, dtype
):
    name = f"awkward_BitMaskedArray_to_{kind}"
    source = np.arange(256, dtype=np.uint8)[:length]
    actual = np.full(length * 8 + 2, -99, dtype=dtype)
    expected = actual.tolist()
    # The YAML definition is the cross-backend contract. Keep this host-only
    # callback test independent of the compiled CPU bit-mask implementation:
    # s390x CI exposed a discrepancy in its LSB-first indexed-option output.
    # tests-cuda retains real compiled CPU/GPU parity checks.
    bitmask_specification[name](expected, source.tolist(), length, validwhen, lsb_order)
    callbacks[name](actual, source, length, validwhen, lsb_order)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))
