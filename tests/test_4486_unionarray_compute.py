# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
import ctypes
from functools import lru_cache
from pathlib import Path
from threading import RLock
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

    queries = []

    def prepared(function):
        def factory(**build):
            def call(*, temp_storage, **arguments):
                if temp_storage is None:
                    queries.append(function.__name__)
                    return 16
                return function(**arguments)

            return call

        return factory

    module = SimpleNamespace(
        **{
            name: getattr(np, name)
            for name in (
                "empty",
                "zeros",
                "asarray",
                "maximum",
                "where",
                "int64",
                "uint8",
            )
        },
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(ptr=123),
            runtime=SimpleNamespace(getDevice=lambda: 0),
            driver=SimpleNamespace(ctxGetCurrent=lambda: 1),
        ),
    )
    namespace = {
        "__name__": "awkward._connect.cuda._unionarray",
        "cp": module,
        "lru_cache": lru_cache,
        "RLock": RLock,
        "_flatten_lock": RLock(),
        "_nested_lock": RLock(),
        "make_op_adapter": lambda op: op,
        "make_unary_transform": lambda **build: transform,
        "make_inclusive_scan": prepared(scan),
        "make_reduce_into": prepared(reduce),
        "queries": queries,
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
    state = namespace["_ArrayState"]
    state.bind = lambda self, array: setattr(self, "array", array)
    state.clear = lambda self: setattr(self, "array", None)
    state.__getitem__ = lambda self, index: self.array[index]
    state.__setitem__ = lambda self, index, value: self.array.__setitem__(index, value)
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
    source = ROOT / "src/awkward/_connect/cuda/_unionarray.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_i64"
    )
    assert ast.unparse(node.decorator_list[0]) == "numba_cuda.jit(device=True)"
    node.decorator_list = []
    namespace = {"__name__": "awkward._connect.cuda._unionarray", "ctypes": ctypes}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    read = namespace["_load_i64"]

    values = np.array([123, -(2**63), 2**63 - 1], dtype=np.int64)
    for i, value in enumerate(values):
        assert read(values.ctypes.data, i) == value


def test_prepared_flatten_rebinding_and_workspace_reuse(kernels):
    module, pointers = kernels
    previous = []
    for address, offset_values in ((100, [2, 4, 4]), (200, [1, 2, 5])):
        pointers[address] = np.array(offset_values, dtype=np.int64)
        tags = np.array([0, 0], dtype=np.int8)
        index = np.array([1, 0], dtype=np.int64)
        table = np.array([address], dtype=np.int64)
        total = np.full(2, 99, dtype=np.int64)
        outtags = np.full(10, 99, dtype=np.int8)
        outindex = np.full(10, 99, dtype=np.int64)
        outoffsets = np.full(4, 99, dtype=np.int64)
        module.flatten_length(total, tags, index, 2, table)
        module.flatten_combine(outtags, outindex, outoffsets, tags, index, 2, table)
        expected = list(range(offset_values[1], offset_values[2])) + list(
            range(offset_values[0], offset_values[1])
        )
        assert total.tolist() == [len(expected), 99]
        assert outindex.tolist() == expected + [99] * (10 - len(expected))
        assert outoffsets.tolist() == [
            0,
            offset_values[2] - offset_values[1],
            len(expected),
            99,
        ]
        previous.append((outindex, outindex.copy()))
    assert module.queries == ["reduce", "scan", "scan"]
    for array, snapshot in previous:
        np.testing.assert_array_equal(array, snapshot)
    for combine, dtype in ((False, np.int64), (True, np.int8)):
        setup = module._flatten_setup(
            0,
            1,
            np.dtype(np.int8).str,
            np.dtype(np.int64).str,
            np.dtype(dtype).str,
            2,
            combine,
        )
        assert all(state.array is None for state in setup.states)


@pytest.mark.parametrize("length", [3, 64, 65, 1024])
@pytest.mark.parametrize("overlap", [False, True])
def test_prepared_nested_rebinding_workspace_and_no_upload(kernels, length, overlap):
    module, _ = kernels
    retained = []

    def forbidden(*args, **kwargs):
        raise AssertionError("nested fill must not upload host parameters")

    module.cp.asarray = forbidden
    module.cp.maximum = forbidden
    module.cp.where = forbidden
    module.cp.zeros = forbidden
    for shift in (0, 1):
        starts = np.arange(length, dtype=np.int64) * (1 if overlap else 4)
        counts = np.full(length, 3 + shift, dtype=np.int64)
        counts[length // 2] = -1
        capacity = int(max(starts + np.maximum(counts, 0))) + 3
        tags = np.full(capacity, 99, dtype=np.int8)
        index = np.full(capacity, 99, dtype=np.int64)
        expected_tags, expected_index = tags.copy(), index.copy()
        rank = 0
        for start, count in zip(starts, counts, strict=True):
            for item in range(start, start + count):
                expected_tags[item], expected_index[item] = 3, rank
                rank += 1
        scratch = np.append(starts, 99)
        module.nestedfill_tags_index(tags, index, scratch, 3, counts, length)
        np.testing.assert_array_equal(tags, expected_tags)
        np.testing.assert_array_equal(index, expected_index)
        np.testing.assert_array_equal(scratch, np.append(starts + counts, 99))
        retained.append((index, expected_index))
    assert module.queries == ([] if length <= 64 else ["scan", "scan", "reduce"])
    setup = module._nested_setup(
        0,
        1,
        tags.dtype.str,
        index.dtype.str,
        scratch.dtype.str,
        counts.dtype.str,
        3,
        length,
    )
    assert all(state.array is None for state in setup.states)
    for actual, expected in retained:
        np.testing.assert_array_equal(actual, expected)


def test_actual_union_proxy_metadata_lifecycle():
    import awkward as ak

    source = Path(ak.__file__).parent / "_connect/cuda/_unionarray.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_ArrayState"
    )
    namespace = {"__name__": "awkward._connect.cuda._unionarray", "np": np}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    state = namespace["_ArrayState"](np.int64)
    assert len(state) == 0 and state.data_ptr() == 0
    state.bind(
        SimpleNamespace(
            __cuda_array_interface__={
                "shape": (3,),
                "strides": None,
                "typestr": np.dtype(np.int64).str,
                "data": (4096, False),
                "version": 3,
            }
        )
    )
    assert len(state) == 3 and state.data_ptr() == 4096
    state.clear()
    assert len(state) == 0 and state.data_ptr() == 0
