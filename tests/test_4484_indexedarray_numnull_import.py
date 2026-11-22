# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Execute numnull's actual body without importing GPU-only compiler packages."""

import ast
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("values", [[], [0, 1, 2], [-1, 0, -2], [-(2**63), -1]])
def test_numnull_uses_its_own_stream_adapter(values, monkeypatch):
    class Stream:
        def __cuda_stream__(self):
            return (0, 123)

    own_module = ModuleType("awkward._connect.cuda._indexedarray")
    own_module._Stream = Stream
    monkeypatch.setitem(sys.modules, own_module.__name__, own_module)
    # No sibling migration module may be imported to provide the stream.
    monkeypatch.setitem(sys.modules, "awkward._connect.cuda._listoffsetarray", None)
    source = Path(ak.__file__).parent / "_connect/cuda/_compute.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "awkward_IndexedArray_numnull"
    ]
    launches = []

    def reduce_into(*, d_in, d_out, op, num_items, h_init, stream):
        assert stream.__cuda_stream__() == (0, 123)
        assert h_init.dtype == np.dtype(np.int64)
        assert num_items == len(values)
        launches.append(1)
        d_out[0] = sum(d_in)

    namespace = {
        "np": np,
        "TransformIterator": lambda array, op: [op(x) for x in array],
        "reduce_into": reduce_into,
        "OpKind": type("OpKind", (), {"PLUS": "plus"}),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    output = np.zeros(1, dtype=np.int64)
    namespace["awkward_IndexedArray_numnull"](
        output, np.asarray(values, dtype=np.int64), len(values)
    )
    assert output[0] == sum(x < 0 for x in values)
    assert launches == [1]
