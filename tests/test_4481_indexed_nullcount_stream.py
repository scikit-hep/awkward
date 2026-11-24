# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("length", [0, 1, 8, 65, 4097])
def test_nullcount_forwards_current_stream(dtype, length):
    path = Path(ak.__file__).resolve().parent / "_connect/cuda/_compute.py"
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "awkward_IndexedArray_numnull"
    ]
    assert len(body) == 1
    stream = SimpleNamespace(ptr=12345)
    calls = []

    def reduce(*, d_in, d_out, op, num_items, h_init, stream):
        assert stream.__cuda_stream__() == (0, 12345)
        assert op == "plus"
        calls.append(num_items)
        d_out[0] = int(h_init[0]) + sum(d_in[:num_items])

    namespace = {
        "__name__": "awkward._connect.cuda._compute",
        "np": np,
        "cp": SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: stream)),
        "SimpleNamespace": SimpleNamespace,
        "OpKind": SimpleNamespace(PLUS="plus"),
        "TransformIterator": lambda values, op: [op(x) for x in values],
        "reduce_into": reduce,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    source = np.arange(length + 3, dtype=dtype)
    if dtype is not np.uint32:
        source[1:length:3] = -1
    actual = np.array([123, 456], dtype=np.int64)
    expected = actual.copy()
    error = NumpyBackend.instance()["awkward_IndexedArray_numnull", np.int64, dtype](
        expected, source, length
    )
    assert not error.str
    namespace["awkward_IndexedArray_numnull"](actual, source, length)
    np.testing.assert_array_equal(actual, expected)
    assert calls == [length]
