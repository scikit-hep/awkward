# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from awkward._backends.numpy import NumpyBackend

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kernels():
    source = ROOT / "src/awkward/_connect/cuda/_utf8.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]

    def transform(**kw):
        assert kw["stream"].__cuda_stream__() == (0, 123)
        for i in range(kw["num_items"]):
            value = kw["op"](i)
            if kw["d_out"] is not None:
                kw["d_out"][i] = value

    def reduce(**kw):
        assert kw["stream"].__cuda_stream__() == (0, 123)
        values = kw["d_in"][: kw["num_items"]]
        kw["d_out"][0] = (
            values.sum()
            if kw["op"] == "sum"
            else (values.max() if kw["op"] == "max" else values.min())
        )

    class DeviceArray(np.ndarray):
        def get(self):
            return np.asarray(self).copy()

    def zeros(*args, **kwargs):
        return np.zeros(*args, **kwargs).view(DeviceArray)

    module = SimpleNamespace(
        **{
            name: getattr(np, name)
            for name in ("empty", "zeros", "asarray", "maximum", "where", "int64")
        },
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=123)),
    )
    module.zeros = zeros
    namespace = {
        "__name__": "awkward._connect.cuda._utf8",
        "cp": module,
        "np": np,
        "numba_cuda": SimpleNamespace(jit=lambda **kwargs: lambda f: f),
        "dataclass": dataclass,
        "OpKind": SimpleNamespace(PLUS="sum", MAXIMUM="max", MINIMUM="min"),
        "CountingIterator": lambda x: x,
        "DiscardIterator": lambda: None,
        "unary_transform": transform,
        "reduce_into": reduce,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


CASES = [
    [],
    [""],
    ["", ""],
    ["a", "", "abc"],
    ["é", "日本", "😀"],
    ["a\x00b", "Ω𐐷"],
    ["\U0010ffff", "\x7f", "\u0080", "\u07ff", "\u0800"],
]


def buffers(strings, dtype, origin):
    data = bytearray(b"x" * origin)
    offsets = [origin]
    for value in strings:
        data.extend(value.encode("utf-8"))
        offsets.append(len(data))
    return np.asarray(data, dtype=np.uint8), np.asarray(offsets, dtype=dtype)


@pytest.mark.parametrize("strings", CASES)
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("origin", [0, 3])
def test_conversion(kernels, strings, dtype, origin):
    data, offsets = buffers(strings, dtype, origin)
    maximum = np.full(3, 99, dtype=np.int64)
    native_maximum = maximum.copy()
    backend = NumpyBackend.instance()
    error = backend[
        "awkward_NumpyArray_prepare_utf8_to_utf32_padded", np.uint8, dtype, np.int64
    ](data, offsets, len(offsets), native_maximum)
    assert error.str is None
    kernels.prepare(data, offsets, len(offsets), maximum)
    np.testing.assert_array_equal(maximum, native_maximum)
    width = int(maximum[0]) + 2
    output = np.full(width * len(strings) + 4, 99, dtype=np.uint32)
    native = output.copy()
    error = backend[
        "awkward_NumpyArray_utf8_to_utf32_padded", np.uint8, dtype, np.uint32
    ](data, offsets, len(offsets), width, native)
    assert error.str is None
    assert kernels.decode(data, offsets, len(offsets), width, output) is None
    np.testing.assert_array_equal(output, native)
    expected = [
        ord(c) for value in strings for c in value + "\x00" * (width - len(value))
    ] + [99] * 4
    assert output.tolist() == expected


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("invalid", [0x80, 0xBF, 0xF8, 0xFF])
@pytest.mark.parametrize("row", [0, 1, 2])
def test_invalid_lead(kernels, dtype, invalid, row):
    data = np.asarray([97, 98, 99, 100, 101, 102], dtype=np.uint8)
    data[row * 2 + 1] = invalid
    offsets = np.asarray([0, 2, 4, 6], dtype=dtype)
    output = np.full(11, 99, dtype=np.uint32)
    native = output.copy()
    error = NumpyBackend.instance()[
        "awkward_NumpyArray_utf8_to_utf32_padded", np.uint8, dtype, np.uint32
    ](data, offsets, len(offsets), 3, native)
    actual = kernels.decode(data, offsets, len(offsets), 3, output)
    assert actual.str == error.str
    assert actual.id == error.id
    assert actual.attempt == error.attempt
    np.testing.assert_array_equal(output, native)


@pytest.mark.parametrize(
    "data,expected",
    [
        ([0xC0, 0x80], [0]),
        ([0xC2, 0x41], [129]),
        ([0xED, 0xA0, 0x80], [0xD800]),
        ([0xF4, 0x90, 0x80, 0x80], [0x110000]),
    ],
)
def test_cpu_byte_semantics(kernels, data, expected):
    data = np.asarray(data, dtype=np.uint8)
    offsets = np.asarray([0, len(data)], dtype=np.int64)
    output, native = np.full(3, 99, dtype=np.uint32), np.full(3, 99, dtype=np.uint32)
    error = NumpyBackend.instance()[
        "awkward_NumpyArray_utf8_to_utf32_padded", np.uint8, np.int64, np.uint32
    ](data, offsets, 2, 2, native)
    assert error.str is None
    assert kernels.decode(data, offsets, 2, 2, output) is None
    np.testing.assert_array_equal(output, native)
    assert output.tolist() == [*expected, 0, 99]


def test_native_width():
    numba = pytest.importorskip("numba")
    source = ROOT / "src/awkward/_connect/cuda/_utf8.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_width"
    ]
    namespace = {"numba_cuda": SimpleNamespace(jit=lambda **kwargs: lambda f: f)}
    exec(compile(tree, str(source), "exec"), namespace)
    compiled = numba.njit(namespace["_width"])
    for byte in range(256):
        assert compiled(np.uint8(byte)) == namespace["_width"](byte)
