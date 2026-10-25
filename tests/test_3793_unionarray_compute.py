# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kernels():
    source = ROOT / "src/awkward/_connect/cuda/_unionarray.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and getattr(n, "name", "") != "_load_i64"
    ]
    pointers = {}

    def transform(**kw):
        assert kw["stream"].__cuda_stream__() == (0, 123)
        for i in range(kw["num_items"]):
            value = kw["op"](i)
            if kw["d_out"] is not None:
                kw["d_out"][i] = value

    def scan(**kw):
        assert kw["stream"].__cuda_stream__() == (0, 123)
        values = kw["d_in"][: kw["num_items"]]
        kw["d_out"][:] = (
            np.cumsum(values) if kw["op"] == "sum" else np.maximum.accumulate(values)
        )

    def reduce(**kw):
        assert kw["stream"].__cuda_stream__() == (0, 123)
        values = kw["d_in"][: kw["num_items"]]
        kw["d_out"][0] = values.sum() if kw["op"] == "sum" else values.max()

    module = SimpleNamespace(
        **{
            name: getattr(np, name)
            for name in ("empty", "zeros", "asarray", "maximum", "where", "int64")
        },
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=123)),
    )
    namespace = {
        "cp": module,
        "np": np,
        "intrinsic": lambda f: f,
        "OpKind": SimpleNamespace(PLUS="sum", MAXIMUM="max"),
        "CountingIterator": lambda x: x,
        "DiscardIterator": lambda: None,
        "unary_transform": transform,
        "inclusive_scan": scan,
        "reduce_into": reduce,
        "_load_i64": lambda address, i: pointers[address][i],
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace), pointers


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("rows", [[], [0], [2, 0, 1, 2, 1]])
@pytest.mark.parametrize(
    "offsets", [[[2, 4, 4, 7]], [[4, 2, 5, 5], [1, 1, 3, 6], [0, 2, 3, 5]]]
)
def test_flatten(kernels, dtype, rows, offsets):
    module, pointers = kernels
    for i, values in enumerate(offsets):
        pointers[i + 100] = np.array(values, dtype=np.int64)
    table = np.array(list(pointers), dtype=np.intp)
    tags = np.array([i % len(offsets) for i in rows], dtype=np.int8)
    index = np.array(rows, dtype=dtype)
    expected_offsets = [0]
    expected_tags, expected_index = [], []
    for tag, idx in zip(tags, index, strict=False):
        start, stop = offsets[tag][idx : idx + 2]
        expected_offsets.append(expected_offsets[-1] + stop - start)
        for j in range(start, stop):
            expected_tags.append(tag)
            expected_index.append(j)
    total = np.full(3, 99, dtype=np.int64)
    module.flatten_length(total, tags, index, len(rows), table)
    assert total.tolist() == [expected_offsets[-1], 99, 99]
    outtags = np.full(50, 99, dtype=np.int8)
    outindex = np.full(50, 99, dtype=np.int64)
    outoffsets = np.full(len(rows) + 3, 99, dtype=np.int64)
    module.flatten_combine(outtags, outindex, outoffsets, tags, index, len(rows), table)
    assert outtags.tolist() == expected_tags + [99] * (50 - len(expected_tags))
    assert outindex.tolist() == expected_index + [99] * (50 - len(expected_index))
    assert outoffsets.tolist() == [*expected_offsets, 99, 99]


@pytest.mark.parametrize(
    "starts,counts",
    [
        ([], []),
        ([0, 3, 8], [2, 0, 3]),
        ([5, 0, 2], [3, 4, 2]),
        ([0, 1, 2], [4, 3, 2]),
        ([4, 4, 7], [-2, 0, 2]),
    ],
)
@pytest.mark.parametrize(
    "tag_dtype,index_dtype",
    [
        (np.int64, np.int64),
        (np.int8, np.int32),
        (np.int8, np.int64),
        (np.int8, np.uint32),
    ],
)
def test_nested(kernels, starts, counts, tag_dtype, index_dtype):
    module, _ = kernels
    tags = np.full(20, 99, dtype=tag_dtype)
    index = np.full(20, 99, dtype=index_dtype)
    expected_tags, expected_index = tags.copy(), index.copy()
    rank = 0
    for start, count in zip(starts, counts, strict=False):
        for j in range(start, start + count):
            expected_tags[j] = 3
            expected_index[j] = rank
            rank += 1
    scratch = np.array([*starts, 99], dtype=np.int64)
    module.nestedfill_tags_index(
        tags, index, scratch, 3, np.array(counts, dtype=np.int64), len(starts)
    )
    np.testing.assert_array_equal(tags, expected_tags)
    np.testing.assert_array_equal(index, expected_index)
    assert scratch.tolist() == [x + y for x, y in zip(starts, counts, strict=False)] + [
        99
    ]


def test_pointer_load_native():
    numba = pytest.importorskip("numba")
    from numba import types
    from numba.extending import intrinsic

    source = ROOT / "src/awkward/_connect/cuda/_unionarray.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_i64"
    ]
    namespace = {"types": types, "intrinsic": intrinsic}
    exec(compile(tree, str(source), "exec"), namespace)
    load = namespace["_load_i64"]

    @numba.njit
    def read(address, index):
        return load(address, index)

    values = np.array([123, -(2**63), 2**63 - 1], dtype=np.int64)
    for i, value in enumerate(values):
        assert read(values.ctypes.data, i) == value
