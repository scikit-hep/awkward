# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU contract checks and host execution of the actual CCCL operator bodies.

Host substitutes do not test GPU compilation or execution; tests-cuda runs the
same case inventory through real CUDA dispatch.
"""

import ast
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
# Execute the installed implementation so coverage attributes host callbacks to
# the same package files as the normal test run (CI uses a wheel install).
CUDA_SOURCE = Path(ak.__file__).resolve().parent / "_connect/cuda"
FUNCTIONS = {
    "awkward_ListArray_validity": "validity",
    "awkward_ListArray_getitem_next_at": "next_at",
    "awkward_ListArray_getitem_next_array": "next_array",
    "awkward_ListArray_getitem_next_array_advanced": "next_array_advanced",
    "awkward_ListArray_getitem_jagged_expand": "jagged_expand",
    "awkward_ListArray_getitem_jagged_apply": "jagged_apply",
    "awkward_ListArray_getitem_jagged_shrink": "jagged_shrink",
    "awkward_ListArray_rpad_axis1": "rpad",
    "awkward_ListArray_rpad_and_clip_length_axis1": "awkward_ListArray_rpad_and_clip_length_axis1",
    "awkward_ListArray_getitem_next_range_spreadadvanced": "spreadadvanced",
    "awkward_ListArray_getitem_next_range": "next_range",
    "awkward_ListArray_getitem_next_range_carrylength": "awkward_ListArray_getitem_next_range_carrylength",
}
DTYPES = {"int32_t": np.int32, "uint32_t": np.uint32, "int64_t": np.int64}


def cases():
    specifications = yaml.safe_load((ROOT / "kernel-specification.yml").read_text())[
        "kernels"
    ]
    data = {
        x["name"]: x["tests"]
        for x in json.loads((ROOT / "kernel-test-data.json").read_text())["unit-tests"]
    }
    output = []
    for spec in specifications:
        if spec["name"] not in FUNCTIONS:
            continue
        for specialization in spec["specializations"]:
            for i, case in enumerate(data[spec["name"]]):
                valid = True
                for argument in specialization["args"]:
                    if "List[" in argument["type"]:
                        dtype = next(
                            v for k, v in DTYPES.items() if k in argument["type"]
                        )
                        values = case["inputs"].get(
                            argument["name"], case["outputs"].get(argument["name"], [])
                        )
                        limits = np.iinfo(dtype)
                        if any(x < limits.min or x > limits.max for x in values):
                            valid = False
                if valid:
                    output.append(
                        pytest.param(
                            spec["name"],
                            specialization,
                            case,
                            id=f"{specialization['name']}-{i}",
                        )
                    )
    return output


CASES = cases()


def arguments(specialization, case):
    values = []
    outputs = []
    key = []
    for argument in specialization["args"]:
        name = argument["name"]
        if "List[" in argument["type"]:
            dtype = next(v for k, v in DTYPES.items() if k in argument["type"])
            key.append(dtype)
            if argument["dir"] == "out":
                sentinel = np.iinfo(dtype).max - 2
                value = np.full(len(case["outputs"][name]) + 8, sentinel, dtype=dtype)
                outputs.append(len(values))
            else:
                value = np.array(case["inputs"][name], dtype=dtype)
            values.append(value)
        else:
            values.append(case["inputs"][name])
    return values, outputs, key


class _DeviceArray(np.ndarray):
    reads = 0

    def get(self):
        type(self).reads += 1
        return np.asarray(self).copy()


class _Counting:
    def __init__(self, start):
        self.start = start
        self.value_type = np.dtype(type(start))


class _Discard:
    pass


class _Transformed:
    def __init__(self, iterator, operation, *, value_type):
        # Match CCCL 1.3's stateful-op limitation instead of inferring a type
        # in host emulation and masking a GPU-only constructor failure.
        assert value_type == iterator.value_type
        self.iterator, self.operation = iterator, operation

    def __getitem__(self, index):
        return [self.operation(self.iterator.start + i) for i in range(index.stop)]


def host_implementation():
    def array_factory(function):
        def allocate(*args, **kwargs):
            return function(*args, **kwargs).view(_DeviceArray)

        return allocate

    cp = SimpleNamespace(
        **{
            name: array_factory(getattr(np, name))
            for name in ("empty", "zeros", "full", "asarray")
        },
        int64=np.int64,
        maximum=np.maximum,
        sum=np.sum,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=123)),
    )

    def transform(*, d_in, d_out, op, num_items, stream):
        assert stream.__cuda_stream__() == (0, 123)
        for i in range(num_items):
            value = op(np.int64(d_in.start + i))
            if not isinstance(d_out, _Discard):
                d_out[i] = value

    def scan(*, d_in, d_out, num_items, **kwargs):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        d_out[:num_items] = np.cumsum(d_in[:num_items])

    def reduce(*, d_in, d_out, num_items, h_init, **kwargs):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        values = [
            int(np.asarray(h_init).reshape(-1)[0]),
            *(int(x) for x in d_in[:num_items]),
        ]
        d_out[0] = sum(values) if kwargs["op"] == 0 else min(values)

    tree = ast.parse((CUDA_SOURCE / "_listarray.py").read_text())
    body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {
        "__name__": "awkward._connect.cuda._listarray",
        "np": np,
        "cp": cp,
        "dataclass": dataclass,
        "SimpleNamespace": SimpleNamespace,
        "_kSliceNone": ak._util.kSliceNone,
        "_SLICE_NONE": ak._util.kSliceNone,
        "CountingIterator": _Counting,
        "TransformIterator": _Transformed,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(PLUS=0, MINIMUM=1),
        "unary_transform": transform,
        "inclusive_scan": scan,
        "reduce_into": reduce,
    }
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(CUDA_SOURCE / "_listarray.py"),
            "exec",
        ),
        namespace,
    )
    source = CUDA_SOURCE / "_compute.py"
    tree = ast.parse(source.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "_make_count_row",
            "awkward_ListArray_getitem_next_range_carrylength",
            "awkward_ListArray_rpad_and_clip_length_axis1",
        )
    ]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace


@pytest.fixture(scope="module")
def implementation():
    return host_implementation()


def assert_error(actual, expected):
    assert (actual is None or actual.str is None) == (
        expected is None or expected.str is None
    )
    if expected is not None and expected.str is not None:
        assert actual.str == expected.str
        assert actual.id == expected.id
        assert actual.attempt == expected.attempt


@pytest.mark.parametrize("name,specialization,case", CASES)
def test_cpu_parity(name, specialization, case, implementation):
    args, outputs, key = arguments(specialization, case)
    expected_args = copy.deepcopy(args)
    expected = NumpyBackend.instance()[(name, *key)](*expected_args)
    assert bool(expected.str) == bool(case["error"])
    _DeviceArray.reads = 0
    actual = implementation[FUNCTIONS[name]](*args)
    assert_error(actual, expected)
    for i in outputs:
        np.testing.assert_array_equal(args[i], expected_args[i])
    assert _DeviceArray.reads <= 1
    if name in {
        "awkward_ListArray_getitem_next_range",
        "awkward_ListArray_getitem_next_range_carrylength",
        "awkward_ListArray_getitem_next_range_spreadadvanced",
        "awkward_ListArray_getitem_jagged_shrink",
        "awkward_ListArray_rpad_axis1",
    }:
        assert _DeviceArray.reads == 0


def test_raw_dispatch_removed():
    tree = ast.parse((ROOT / "dev/generate-kernel-signatures.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "cuda_kernels_impl" for t in n.targets
        )
    )
    assert FUNCTIONS.keys().isdisjoint(ast.literal_eval(node.value))


@pytest.fixture(scope="module")
def compute_backend(implementation):
    class ComputeOnHost(NumpyBackend):
        def _new_kernel(self, key):
            if key[0] in FUNCTIONS:
                return implementation[FUNCTIONS[key[0]]]
            return super()._new_kernel(key)

    backend = object.__new__(ComputeOnHost)
    backend.__init__()
    return backend


@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("exception", [False, True])
def test_public_validity_contract(compute_backend, offsets, exception):
    content = ak.contents.NumpyArray(np.array([1]))
    if offsets:
        layout = ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2])), content
        )
    else:
        layout = ak.contents.ListArray(
            ak.index.Index64(np.array([0])), ak.index.Index64(np.array([2])), content
        )
    layout._backend = compute_backend
    if exception:
        with pytest.raises(ValueError, match="stop"):
            ak.validity_error(layout, exception=True)
        with pytest.raises(ValueError, match="stop"):
            ak.is_valid(layout, exception=True)
    else:
        assert "stop" in ak.validity_error(layout)
        assert ak.is_valid(layout) is False


@pytest.mark.parametrize("selector", [0, [0, 99], [[0], [0], [0]]])
def test_public_index_error_contract(compute_backend, selector):
    layout = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 2, 2])),
        ak.index.Index64(np.array([2, 2, 4])),
        ak.contents.NumpyArray(np.arange(4)),
    )
    layout._backend = compute_backend
    with pytest.raises(IndexError):
        if isinstance(selector, list) and isinstance(selector[0], list):
            ak.Array(layout)[selector]
        else:
            ak.Array(layout)[:, selector]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_public_slicing_operations(compute_backend, dtype):
    content = ak.contents.NumpyArray(np.arange(10))
    layout = ak.contents.ListArray(
        ak.index.Index(np.asarray([0, 4, 2, 7], dtype=dtype)),
        ak.index.Index(np.asarray([3, 4, 5, 9], dtype=dtype)),
        content,
    )
    source = ak.Array(layout)
    slices = [slice(None, None, -1), slice(1, None), slice(None, None, 2)]
    jagged = [
        [[1, 0], [], [-1, 0], [1]],
        [[0, None, -1], [], [None, 1], [0, 1, None]],
    ]
    expected_slices = [source[:, index].to_list() for index in slices]
    expected_jagged = [source[index].to_list() for index in jagged]
    expected_pad = ak.pad_none(source, 4).to_list()
    layout._backend = compute_backend
    content._backend = compute_backend
    for index, expected in zip(slices, expected_slices, strict=True):
        assert source[:, index].to_list() == expected
    for index, expected in zip(jagged, expected_jagged, strict=True):
        assert source[index].to_list() == expected
    assert ak.pad_none(source, 4).to_list() == expected_pad


@pytest.fixture(scope="module")
def prepared_implementation():
    import weakref
    from functools import lru_cache
    from threading import RLock

    registry = weakref.WeakValueDictionary()
    events = []

    class Buffer(_DeviceArray):
        @property
        def __cuda_array_interface__(self):
            registry[self.ctypes.data] = self
            return {
                "shape": self.shape,
                "strides": None,
                "typestr": self.dtype.str,
                "data": (self.ctypes.data, False),
                "version": 3,
            }

    def prepare(**kwargs):
        events.append(("prepare",))
        operation = kwargs["op"]

        def launch(**kwargs):
            import time

            time.sleep(0.0001)
            events.append(("transform", kwargs["num_items"]))
            for i in range(kwargs["num_items"]):
                value = operation(i)
                if not isinstance(kwargs["d_out"], _Discard):
                    kwargs["d_out"][i] = value

        return launch

    def prepare_reduce(**kwargs):
        events.append(("prepare_reduce",))

        def launch(*, temp_storage, **kwargs):
            if temp_storage is None:
                events.append(("reduce_storage_query",))
                return 64
            events.append(("reduce", kwargs["num_items"]))
            source = kwargs["d_in"]
            items = kwargs["num_items"]
            values = (
                [source.start + i for i in range(items)]
                if isinstance(source, _Counting)
                else source[:items]
            )
            if callable(kwargs["op"]):
                from functools import reduce

                result = reduce(kwargs["op"], values, int(kwargs["h_init"]))
            else:
                values = [int(kwargs["h_init"]), *values]
                result = max(values) if kwargs["op"] == 2 else min(values)
            kwargs["d_out"][0] = result
            return 64

        return launch

    def empty(length, dtype):
        events.append(("allocate", length, np.dtype(dtype).str))
        return np.empty(length, dtype=dtype).view(Buffer)

    namespace = {
        "np": np,
        "cp": SimpleNamespace(
            empty=empty,
            asarray=lambda values, dtype: np.asarray(values, dtype=dtype).view(Buffer),
            int64=np.int64,
            uint8=np.uint8,
            cuda=SimpleNamespace(
                get_current_stream=lambda: SimpleNamespace(ptr=123),
                runtime=SimpleNamespace(getDevice=lambda: 0),
                driver=SimpleNamespace(ctxGetCurrent=lambda: 1),
            ),
        ),
        "lru_cache": lru_cache,
        "_lock": RLock(),
        "CountingIterator": _Counting,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(MINIMUM=1, MAXIMUM=2),
        "make_op_adapter": lambda op: op,
        "make_unary_transform": prepare,
        "make_reduce_into": prepare_reduce,
    }
    namespace["__name__"] = "awkward._connect.cuda._listarray_prepared"
    path = CUDA_SOURCE / "_listarray_prepared.py"
    tree = ast.parse(path.read_text())
    body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    state = namespace["_ArrayState"]
    state.__getitem__ = lambda self, i: registry[self.data_ptr()][i]
    state.__setitem__ = lambda self, i, value: registry[self.data_ptr()].__setitem__(
        i, value
    )
    return namespace, Buffer, events


PREPARED_NAMES = {
    "awkward_ListArray_getitem_next_array": "array",
    "awkward_ListArray_getitem_next_array_advanced": "advanced",
    "awkward_ListArray_getitem_jagged_expand": "jagged",
    "awkward_ListArray_getitem_next_range_spreadadvanced": "spread",
}


def run_prepared(namespace, implementation, name, args, **options):
    kind = PREPARED_NAMES[name]
    if kind == "spread":
        length, parameter, arrays = args[3], 0, args[:3]
    elif kind == "jagged":
        length, parameter, arrays = args[7], args[6], args[:6]
    elif kind == "array":
        length, parameter = args[5], args[7]
        arrays = (*args[:2], args[2][:length], args[3][:length], args[4][: args[6]])
    else:
        length, parameter, arrays = args[6], args[7], args[:6]
    if not length:
        return None
    status = namespace["run"](
        kind, tuple(arrays), length, parameter, implementation["_Stream"](), **options
    )
    if kind == "spread":
        return None
    if kind == "array":
        return implementation["_array_error"](status)
    messages = (
        ("stops[i] < starts[i]", "cannot fit jagged slice into nested list")
        if kind == "jagged"
        else ("stops[i] < starts[i]", "stops[i] > len(content)", "index out of range")
    )
    return implementation["_prepared_error"](status, length, messages, name)


@pytest.mark.parametrize(
    "name,specialization,case",
    [case for case in CASES if case.values[0] in PREPARED_NAMES],
)
def test_prepared_cpu_parity(
    name, specialization, case, prepared_implementation, implementation
):
    namespace, buffer, _ = prepared_implementation
    args, outputs, key = arguments(specialization, case)
    expected_args = copy.deepcopy(args)
    expected = NumpyBackend.instance()[(name, *key)](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_prepared(namespace, implementation, name, device)
    assert_error(actual, expected)
    for i in outputs:
        np.testing.assert_array_equal(device[i], expected_args[i])


@pytest.mark.parametrize("rows,width", [(32, 8), (32, 4096), (65, 8), (257, 33)])
@pytest.mark.parametrize("error_at", [None, 0, 31, 64])
@pytest.mark.parametrize("chunk_size", [1, 8, 64])
def test_prepared_jagged_parallel_prefix(
    rows, width, error_at, chunk_size, prepared_implementation, implementation
):
    namespace, buffer, events = prepared_implementation
    starts = np.arange(rows, dtype=np.int64) * width
    stops = starts + width
    if error_at is not None and error_at < rows:
        stops[error_at] -= 1
        stops[-1] = starts[-1] - 1
    output = np.full(rows * width + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        output.copy(),
        np.arange(width + 1, dtype=np.int64),
        output.copy(),
        starts,
        stops,
        width,
        rows,
    ]
    expected = copy.deepcopy(args)
    name = "awkward_ListArray_getitem_jagged_expand"
    error = NumpyBackend.instance()[(name, *([np.int64] * 6))](*expected)
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    events.clear()
    actual = run_prepared(
        namespace, implementation, name, device, chunk_size=chunk_size
    )
    assert_error(actual, error)
    for i in (0, 1, 3):
        np.testing.assert_array_equal(device[i], expected[i])
    assert ("transform", rows * width) in events


@pytest.mark.parametrize(
    "offsets",
    [
        [3, 3, 5, 6],
        [0, 8, 2, 10],
        [5, 2, 7, 4, 9],
        [0, 4096, 8192],
        list(range(66)),
        [*range(33), 5, *range(34, 66)],
    ],
)
def test_prepared_spread_sequential_overlap(
    offsets, prepared_implementation, implementation
):
    namespace, buffer, _ = prepared_implementation
    length = len(offsets) - 1
    output = np.full(max(offsets) + 3, 123, dtype=np.int64)
    args = [
        output,
        np.arange(length, dtype=np.int64) + 7,
        np.asarray(offsets, dtype=np.int64),
        length,
    ]
    expected = output.copy()
    NumpyBackend.instance()[
        "awkward_ListArray_getitem_next_range_spreadadvanced",
        np.int64,
        np.int64,
        np.int64,
    ](expected, *args[1:])
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    run_prepared(
        namespace,
        implementation,
        "awkward_ListArray_getitem_next_range_spreadadvanced",
        device,
    )
    np.testing.assert_array_equal(device[0], expected)


def test_prepared_reuse_lifetimes_and_concurrent_stream_state(
    prepared_implementation, implementation
):
    import gc
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    namespace, buffer, events = prepared_implementation

    def run_case(task):
        kind, seed = task
        output = np.full(15, 123, dtype=np.int64)
        starts = np.arange(3, dtype=np.int64) * 4 + seed % 4
        if kind == "jagged":
            name = "awkward_ListArray_getitem_jagged_expand"
            args = [
                output.copy(),
                output.copy(),
                np.arange(5, dtype=np.int64) * (seed + 1),
                output.copy(),
                starts,
                starts + 4,
                4,
                3,
            ]
        elif kind == "advanced":
            name = "awkward_ListArray_getitem_next_array_advanced"
            args = [
                output.copy(),
                output.copy(),
                starts,
                starts + 4,
                np.asarray([0, 10 if seed % 2 else -1], dtype=np.int64),
                np.asarray([0, 1, 0], dtype=np.int64),
                3,
                64,
            ]
        else:
            name = "awkward_ListArray_getitem_next_range_spreadadvanced"
            args = [
                output.copy(),
                np.arange(3, dtype=np.int64) + seed,
                np.asarray(
                    [0, 8, 2, 12] if seed % 2 else [0, 4, 8, 12], dtype=np.int64
                ),
                3,
            ]
        expected = copy.deepcopy(args)
        key = tuple(x.dtype.type for x in args if isinstance(x, np.ndarray))
        error = NumpyBackend.instance()[(name, *key)](*expected)
        device = [
            x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args
        ]
        references = [weakref.ref(x) for x in device if isinstance(x, np.ndarray)]
        actual = run_prepared(namespace, implementation, name, device)
        assert_error(actual, error)
        output_indices = (
            (0, 1, 3) if kind == "jagged" else ((0, 1) if kind == "advanced" else (0,))
        )
        for i in output_indices:
            np.testing.assert_array_equal(device[i], expected[i])
        return references

    tasks = [
        (kind, seed) for seed in range(4) for kind in ("jagged", "advanced", "spread")
    ]
    for kind in ("jagged", "advanced", "spread"):
        run_case((kind, 0))
    events.clear()
    with ThreadPoolExecutor(max_workers=4) as executor:
        references = list(executor.map(run_case, tasks))
    gc.collect()
    assert all(reference() is None for group in references for reference in group)
    assert not any(event[0].startswith("prepare") for event in events)


def test_small_prepared_paths_use_one_launch(prepared_implementation, implementation):
    namespace, buffer, events = prepared_implementation
    output = np.full(19, 123, dtype=np.int64)
    cases = {
        "awkward_ListArray_getitem_next_array_advanced": [
            output.copy(),
            output.copy(),
            np.asarray([0, 4, 8], dtype=np.int64),
            np.asarray([4, 8, 12], dtype=np.int64),
            np.asarray([0, -1], dtype=np.int64),
            np.asarray([0, 1, 0], dtype=np.int64),
            3,
            12,
        ],
        "awkward_ListArray_getitem_next_range_spreadadvanced": [
            output.copy(),
            np.asarray([7, 8, 9], dtype=np.int64),
            np.asarray([0, 8, 2, 12], dtype=np.int64),
            3,
        ],
    }
    for name, args in cases.items():
        expected = copy.deepcopy(args)
        key = tuple(x.dtype.type for x in args if isinstance(x, np.ndarray))
        error = NumpyBackend.instance()[(name, *key)](*expected)
        device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
        events.clear()
        actual = run_prepared(namespace, implementation, name, device)
        assert_error(actual, error)
        transforms = [event for event in events if event[0] == "transform"]
        expected_items = 1 if "array_advanced" in name else output.size
        assert transforms == [("transform", expected_items)]
        allocations = [event for event in events if event[0] == "allocate"]
        assert allocations == (
            []
            if "spreadadvanced" in name
            else [("allocate", 2, np.dtype(np.int64).str)]
        )
        np.testing.assert_array_equal(device[0], expected[0])


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("error_at", [None, 0, 63, 64, 128])
def test_tunable_advanced_error_order(
    chunk_size, error_at, prepared_implementation, implementation
):
    namespace, buffer, _ = prepared_implementation
    rows = 129
    starts = np.arange(rows, dtype=np.int64) * 8
    stops = starts + 8
    indices = np.asarray([0, -1, 8], dtype=np.int64)
    previous = np.arange(rows, dtype=np.int64) % 2
    if error_at is not None:
        previous[error_at] = 2
        if error_at < rows - 1:
            stops[-1] = starts[-1] - 1
    args = [
        np.full(rows + 3, 123, dtype=np.int64),
        np.full(rows + 3, 123, dtype=np.int64),
        starts,
        stops,
        indices,
        previous,
        rows,
        rows * 8,
    ]
    expected = copy.deepcopy(args)
    name = "awkward_ListArray_getitem_next_array_advanced"
    error = NumpyBackend.instance()[(name, *([np.int64] * 6))](*expected)
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_prepared(
        namespace, implementation, name, device, chunk_size=chunk_size
    )
    assert_error(actual, error)
    for i in (0, 1):
        np.testing.assert_array_equal(device[i], expected[i])


@pytest.mark.parametrize(
    "offsets", [[3, 3, 5, 6], [0, 8, 2, 10], [0, 4096, 8192], list(range(66))]
)
def test_tiled_spread_candidate_preserves_cpu_result(
    offsets, prepared_implementation, implementation
):
    namespace, buffer, _ = prepared_implementation
    output = np.full(max(offsets) + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        np.arange(len(offsets) - 1, dtype=np.int64) + 7,
        np.asarray(offsets, dtype=np.int64),
        len(offsets) - 1,
    ]
    expected = output.copy()
    name = "awkward_ListArray_getitem_next_range_spreadadvanced"
    NumpyBackend.instance()[(name, np.int64, np.int64, np.int64)](expected, *args[1:])
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    run_prepared(namespace, implementation, name, device, spread_lanes=32)
    np.testing.assert_array_equal(device[0], expected)


@pytest.mark.parametrize("width,fused", [(8, True), (9, False), (4096, False)])
def test_jagged_fusion_is_bounded(
    width, fused, prepared_implementation, implementation
):
    namespace, buffer, events = prepared_implementation
    rows = 32
    output = np.full(rows * width + 3, 123, dtype=np.int64)
    starts = np.arange(rows, dtype=np.int64) * width
    args = [
        output.copy(),
        output.copy(),
        np.arange(width + 1, dtype=np.int64),
        output.copy(),
        starts,
        starts + width,
        width,
        rows,
    ]
    expected = copy.deepcopy(args)
    name = "awkward_ListArray_getitem_jagged_expand"
    error = NumpyBackend.instance()[(name, *([np.int64] * 6))](*expected)
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    events.clear()
    actual = run_prepared(namespace, implementation, name, device)
    assert_error(actual, error)
    transforms = [event for event in events if event[0] == "transform"]
    assert len(transforms) == (1 if fused else 2)
    for i in (0, 1, 3):
        np.testing.assert_array_equal(device[i], expected[i])


@pytest.fixture(scope="module")
def slicing_implementation(prepared_implementation):
    base, buffer, events = prepared_implementation
    namespace = dict(base)

    def prepare_scan(**kwargs):
        events.append(("prepare_scan",))

        def launch(*, temp_storage, **kwargs):
            if temp_storage is None:
                return 64
            events.append(("scan", kwargs["num_items"]))
            kwargs["d_out"][: kwargs["num_items"]] = np.cumsum(
                kwargs["d_in"][: kwargs["num_items"]]
            )
            return 64

        return launch

    namespace["OpKind"] = SimpleNamespace(MINIMUM=1, MAXIMUM=2, PLUS=0)
    namespace["make_inclusive_scan"] = prepare_scan
    namespace["__name__"] = "awkward._connect.cuda._listarray_slicing"
    path = CUDA_SOURCE / "_listarray_slicing.py"
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, buffer, events


def run_slicing(namespace, implementation, name, args):
    if name == "awkward_ListArray_getitem_jagged_apply":
        arrays = tuple(args[i] for i in (0, 1, 2, 3, 5, 7, 8))
        if not args[4]:
            args[0][0] = 0
            return None
        status = namespace["run"](
            "apply", arrays, args[4], (args[6], args[9]), implementation["_Stream"]()
        )
        return implementation["_apply_error"](status)
    if not args[4]:
        args[0][0] = 0
        return None
    return namespace["run"](
        "range", tuple(args[:4]), args[4], tuple(args[5:]), implementation["_Stream"]()
    )


SLICING_NAMES = {
    "awkward_ListArray_getitem_jagged_apply",
    "awkward_ListArray_getitem_next_range",
}


@pytest.mark.parametrize(
    "name,specialization,case", [c for c in CASES if c.values[0] in SLICING_NAMES]
)
def test_prepared_slicing_contract(
    name, specialization, case, slicing_implementation, implementation
):
    namespace, buffer, _ = slicing_implementation
    args, outputs, key = arguments(specialization, case)
    expected_args = copy.deepcopy(args)
    expected = NumpyBackend.instance()[(name, *key)](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    buffer.reads = 0
    actual = run_slicing(namespace, implementation, name, device)
    assert_error(actual, expected)
    for index in outputs:
        np.testing.assert_array_equal(device[index], expected_args[index])
    assert buffer.reads == (1 if name.endswith("jagged_apply") and args[4] else 0)


def slicing_apply_arguments(
    rows, width, error_at=None, position=1, failure="index", dtype=np.int64
):
    starts = (np.arange(rows) * width).astype(dtype)
    stops = (starts + width).astype(dtype)
    ss = np.arange(rows, dtype=np.int64) * 3
    se = ss + 3
    indices = np.tile(np.array([0, -1, 1], dtype=np.int64), rows)
    if error_at is not None:
        if failure == "index":
            indices[ss[error_at] + position] = width
        elif failure == "slice_stop":
            se[error_at] = len(indices) + 1
        elif failure == "slice_reverse":
            se[error_at] = ss[error_at] - 1
        elif failure == "reverse":
            stops[error_at] = starts[error_at] - 1
        elif failure == "content":
            stops[error_at] = rows * width + 1
    # A later, different error must never win the first-error reduction.
    if error_at is not None and error_at + 1 < rows:
        se[-1] = len(indices) + 2
    return [
        np.full(rows + 4, 123, dtype=np.int64),
        np.full(rows * 3 + 4, 123, dtype=np.int64),
        ss,
        se,
        rows,
        indices,
        len(indices),
        starts,
        stops,
        rows * width,
    ]


@pytest.mark.parametrize("rows", [1, 63, 64, 65, 129, 4097])
@pytest.mark.parametrize(
    "failure", [None, "index", "slice_stop", "slice_reverse", "reverse", "content"]
)
def test_prepared_apply_first_error_and_partial_writes(
    rows, failure, slicing_implementation, implementation
):
    namespace, buffer, _ = slicing_implementation
    args = slicing_apply_arguments(
        rows, 4, min(64, rows - 1) if failure else None, failure=failure
    )
    expected_args = copy.deepcopy(args)
    key = ("awkward_ListArray_getitem_jagged_apply", *([np.int64] * 7))
    expected = NumpyBackend.instance()[key](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_slicing(namespace, implementation, key[0], device)
    assert_error(actual, expected)
    for index in (0, 1):
        np.testing.assert_array_equal(device[index], expected_args[index])


@pytest.mark.parametrize("ss,se,inner", [([0], [0], 0), ([2], [1], 0), ([0], [2], 1)])
def test_apply_does_not_read_skipped_buffers(
    ss, se, inner, slicing_implementation, implementation
):
    namespace, buffer, _ = slicing_implementation
    # The reference never dereferences source bounds for empty or invalid slices.
    args = [
        np.full(3, 123, dtype=np.int64),
        np.full(4, 123, dtype=np.int64),
        np.array(ss, dtype=np.int64),
        np.array(se, dtype=np.int64),
        1,
        np.empty(0, dtype=np.int64),
        inner,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        0,
    ]
    expected_args = copy.deepcopy(args)
    key = ("awkward_ListArray_getitem_jagged_apply", *([np.int64] * 7))
    expected = NumpyBackend.instance()[key](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_slicing(namespace, implementation, key[0], device)
    assert_error(actual, expected)
    for index in (0, 1):
        np.testing.assert_array_equal(device[index], expected_args[index])


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize(
    "start,stop,step",
    [
        (ak._util.kSliceNone, ak._util.kSliceNone, 1),
        (ak._util.kSliceNone, ak._util.kSliceNone, -1),
        (-4, 7, 2),
        (7, -8, -2),
        (99, 2, 1),
        (-99, 99, 3),
        (ak._util.kSliceNone, ak._util.kSliceNone, -9223372036854775808),
    ],
)
def test_prepared_ranges_parallel_output_and_empty_lists(
    dtype, start, stop, step, slicing_implementation, implementation
):
    namespace, buffer, events = slicing_implementation
    starts = np.asarray([2, 11, 5, 4, 2], dtype=dtype)
    stops = np.asarray([10, 11, 4101, 3, 8], dtype=dtype)
    args = [
        np.full(9, 123, dtype=dtype),
        np.full(4200, 123, dtype=np.int64),
        starts,
        stops,
        5,
        start,
        stop,
        step,
    ]
    expected_args = copy.deepcopy(args)
    key = ("awkward_ListArray_getitem_next_range", dtype, np.int64, dtype, dtype)
    expected = NumpyBackend.instance()[key](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    before = len(events)
    buffer.reads = 0
    actual = run_slicing(namespace, implementation, key[0], device)
    assert_error(actual, expected)
    assert buffer.reads == 0
    for index in (0, 1):
        np.testing.assert_array_equal(device[index], expected_args[index])
    assert any(e[0] == "transform" and e[1] >= len(args[1]) for e in events[before:])


def test_slicing_setup_reuse_owners_and_concurrent_streams(
    slicing_implementation, implementation
):
    import gc
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    namespace, buffer, events = slicing_implementation

    def task(pair):
        kind, seed = pair
        if kind == "apply":
            name = "awkward_ListArray_getitem_jagged_apply"
            args = slicing_apply_arguments(129, 4, 64 if seed % 2 else None)
            args[7] += seed
            args[8] += seed
            args[9] = 4096
        else:
            name = "awkward_ListArray_getitem_next_range"
            starts = np.arange(129, dtype=np.int64) * 4 + seed
            args = [
                np.full(133, 123, dtype=np.int64),
                np.full(520, 123, dtype=np.int64),
                starts,
                starts + 4,
                129,
                ak._util.kSliceNone,
                ak._util.kSliceNone,
                -1,
            ]
        expected_args = copy.deepcopy(args)
        key = (name, *(x.dtype.type for x in args if isinstance(x, np.ndarray)))
        expected = NumpyBackend.instance()[key](*expected_args)
        device = [
            x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args
        ]
        refs = [weakref.ref(x) for x in device if isinstance(x, np.ndarray)]
        actual = run_slicing(namespace, implementation, name, device)
        assert_error(actual, expected)
        for i in (0, 1):
            np.testing.assert_array_equal(device[i], expected_args[i])
        return refs

    for kind in ("apply", "range"):
        task((kind, 0))
    events.clear()
    with ThreadPoolExecutor(max_workers=4) as executor:
        groups = list(
            executor.map(
                task, [(kind, seed) for seed in range(4) for kind in ("apply", "range")]
            )
        )
    gc.collect()
    assert all(ref() is None for group in groups for ref in group)
    assert not any(e[0].startswith("prepare") for e in events)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [-9223372036854775808, -5, 4, 9223372036854775807])
def test_apply_preserves_signed_attempts(
    position, bad, slicing_implementation, implementation
):
    namespace, buffer, _ = slicing_implementation
    args = slicing_apply_arguments(65, 4)
    args[5][64 * 3 + position] = bad
    expected_args = copy.deepcopy(args)
    key = ("awkward_ListArray_getitem_jagged_apply", *([np.int64] * 7))
    expected = NumpyBackend.instance()[key](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_slicing(namespace, implementation, key[0], device)
    assert_error(actual, expected)
    for i in (0, 1):
        np.testing.assert_array_equal(device[i], expected_args[i])


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_prepared_range_zero_capacity(dtype, slicing_implementation, implementation):
    namespace, buffer, _ = slicing_implementation
    args = [
        np.full(7, 123, dtype=dtype),
        np.empty(0, dtype=np.int64),
        np.array([0, 3, 4], dtype=dtype),
        np.array([0, 3, 4], dtype=dtype),
        3,
        ak._util.kSliceNone,
        ak._util.kSliceNone,
        -1,
    ]
    expected_args = copy.deepcopy(args)
    key = ("awkward_ListArray_getitem_next_range", dtype, np.int64, dtype, dtype)
    expected = NumpyBackend.instance()[key](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_slicing(namespace, implementation, key[0], device)
    assert_error(actual, expected)
    np.testing.assert_array_equal(device[0], expected_args[0])


def test_jagged_scratch_arena_and_cached_workspace_query(
    prepared_implementation, implementation
):
    namespace, buffer, events = prepared_implementation
    namespace["_setup"].cache_clear()
    rows, width = 131, 2
    starts = np.arange(rows, dtype=np.int64) * width
    output = np.full(rows * width + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        output.copy(),
        np.arange(width + 1, dtype=np.int64),
        output.copy(),
        starts,
        starts + width,
        width,
        rows,
    ]
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    name = "awkward_ListArray_getitem_jagged_expand"
    for iteration in range(2):
        events.clear()
        result = run_prepared(namespace, implementation, name, device)
        assert result is None
        allocations = [event for event in events if event[0] == "allocate"]
        assert allocations == [
            ("allocate", 6, np.dtype(np.int64).str),
            ("allocate", 64, np.dtype(np.uint8).str),
        ]
        queries = [event for event in events if event[0] == "reduce_storage_query"]
        assert len(queries) == (1 if iteration == 0 else 0)
        np.testing.assert_array_equal(device[3][:-3], np.arange(rows * width))
        np.testing.assert_array_equal(device[3][-3:], [123, 123, 123])


@pytest.mark.parametrize("error_kind", ["decreasing", "past_content"])
def test_prepared_advanced_structural_error_after_chunk_boundary(
    error_kind, prepared_implementation, implementation
):
    namespace, buffer, _ = prepared_implementation
    rows, boundary = 129, 65
    starts = np.arange(rows, dtype=np.int64)
    stops = starts + 1
    stops[boundary] = starts[boundary] - 1 if error_kind == "decreasing" else rows + 1
    args = [
        np.full(rows + 2, 123, dtype=np.int64),
        np.full(rows + 2, 123, dtype=np.int64),
        starts,
        stops,
        np.array([0], dtype=np.int64),
        np.zeros(rows, dtype=np.int64),
        rows,
        rows,
    ]
    name = "awkward_ListArray_getitem_next_array_advanced"
    expected = copy.deepcopy(args)
    error = NumpyBackend.instance()[(name, *([np.int64] * 6))](*expected)
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = run_prepared(namespace, implementation, name, device)
    assert_error(actual, error)
    for i in (0, 1):
        np.testing.assert_array_equal(device[i], expected[i])


@pytest.mark.parametrize(
    "name,specialization,case",
    [
        case
        for case in CASES
        if case.values[0] in PREPARED_NAMES
        or case.values[0]
        in (
            "awkward_ListArray_getitem_jagged_apply",
            "awkward_ListArray_getitem_next_range",
        )
    ],
)
def test_prepared_production_wrapper_dispatch(
    name,
    specialization,
    case,
    prepared_implementation,
    slicing_implementation,
    implementation,
    monkeypatch,
):
    import sys
    from types import ModuleType

    namespace, buffer, _ = prepared_implementation
    slicing, _, _ = slicing_implementation
    for suffix, functions in (
        ("_listarray_prepared", namespace),
        ("_listarray_slicing", slicing),
    ):
        module_name = f"awkward._connect.cuda.{suffix}"
        module = ModuleType(module_name)
        module.run = functions["run"]
        monkeypatch.setitem(sys.modules, module_name, module)
    args, outputs, key = arguments(specialization, case)
    expected = copy.deepcopy(args)
    error = NumpyBackend.instance()[(name, *key)](*expected)
    device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    actual = implementation[FUNCTIONS[name]](*device)
    assert_error(actual, error)
    for i in outputs:
        np.testing.assert_array_equal(device[i], expected[i])


def test_prepared_invalid_tuning_parameters(prepared_implementation, implementation):
    namespace, _, _ = prepared_implementation
    for options in ({"chunk_size": 0}, {"spread_lanes": -1}):
        with pytest.raises(ValueError, match="chunk size"):
            namespace["run"]("jagged", (), 1, 1, implementation["_Stream"](), **options)


def test_prepared_descriptor_rebinding_and_clear(prepared_implementation):
    namespace, buffer, _ = prepared_implementation
    state = namespace["_ArrayState"](np.int64)
    assert len(state) == 0
    first = np.arange(3, dtype=np.int64).view(buffer)
    second = np.arange(7, dtype=np.int64).view(buffer)
    state.bind(first)
    assert len(state) == 3
    assert state.data_ptr() == first.ctypes.data
    state.bind(second)
    assert len(state) == 7
    assert state.data_ptr() == second.ctypes.data
    state.clear()
    assert len(state) == 0
    assert state.data_ptr() == 0


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_padding_decreasing_offsets_preserves_guard_buffers(dtype, implementation):
    # A view makes writes before/after the declared output observable.
    for backend in ("cpu", "callbacks"):
        guard = np.full(8, 123, dtype=np.int64)
        starts = np.array([5], dtype=dtype)
        stops = np.array([2], dtype=dtype)
        outstarts, outstops = np.full(1, 123, dtype=dtype), np.full(1, 123, dtype=dtype)
        args = (guard[3:5], starts, stops, outstarts, outstops, 2, 1)
        if backend == "cpu":
            error = NumpyBackend.instance()[
                ("awkward_ListArray_rpad_axis1", np.int64, dtype, dtype, dtype, dtype)
            ](*args)
            assert error.str is None
        else:
            implementation["rpad"](*args)
        np.testing.assert_array_equal(guard, [123, 123, 123, -1, -1, 123, 123, 123])
        np.testing.assert_array_equal(outstarts, [0])
        np.testing.assert_array_equal(outstops, [2])


def test_returned_error_uses_public_formatter(implementation):
    error = implementation["next_at"](
        np.full(1, 123), np.array([0]), np.array([2]), 1, 5
    )
    message = NumpyBackend.instance().format_kernel_error(error)
    assert message == (
        "index out of range while attempting to get index 5 "
        "(in compiled code: cuda.compute: awkward_ListArray_getitem_next_at)"
    )


def test_prepared_setup_reused_across_lengths_and_values(
    prepared_implementation, implementation
):
    namespace, buffer, events = prepared_implementation
    namespace["_setup"].cache_clear()
    prepare_counts = []
    for rows, width in [(2, 3), (129, 4), (4, 5), (129, 4)]:
        starts = np.arange(rows, dtype=np.int64) * width
        output = np.full(rows * width + 3, 123, dtype=np.int64)
        args = [
            output.copy(),
            output.copy(),
            np.arange(width + 1, dtype=np.int64),
            output.copy(),
            starts,
            starts + width,
            width,
            rows,
        ]
        device = [x.view(buffer) if isinstance(x, np.ndarray) else x for x in args]
        assert (
            run_prepared(
                namespace,
                implementation,
                "awkward_ListArray_getitem_jagged_expand",
                device,
            )
            is None
        )
        np.testing.assert_array_equal(
            device[3][: rows * width], np.arange(rows * width)
        )
        prepare_counts.append(sum(event[0] == "prepare" for event in events))
    info = namespace["_setup"].cache_info()
    assert (info.misses, info.hits) == (1, 3)
    # Reusing a previously visited length needs no transform preparation.
    assert prepare_counts[-1] == prepare_counts[-2]


def test_device_closures_do_not_capture_runtime_integers():
    for filename in ("_listarray_prepared.py", "_listarray_slicing.py"):
        tree = ast.parse((CUDA_SOURCE / filename).read_text())
        dynamic = {
            "length",
            "lencontent",
            "inner_length",
            "content_length",
            "start",
            "stop",
            "step",
            "size",
            "capacity",
        }
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            for init in (
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ):
                for op in (
                    node for node in init.body if isinstance(node, ast.FunctionDef)
                ):
                    loads = {
                        node.id
                        for node in ast.walk(op)
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    }
                    stores = {
                        node.id
                        for node in ast.walk(op)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                    }
                    assert not (loads - stores) & dynamic, (filename, cls.name, op.name)


def test_generated_adapter_checks_actual_public_error_format(implementation):
    import re
    import runpy

    generator = runpy.run_path(str(ROOT / "dev/generate-tests.py"))
    namespace = {"re": re, "cupy_backend": NumpyBackend.instance()}
    exec(generator["CUDA_CHECK_ERROR_HELPER"], namespace)
    error = implementation["next_at"](
        np.full(1, 123), np.array([0]), np.array([2]), 1, 5
    )
    checked = namespace["_checked_kernel"](lambda: error)
    name = "awkward_ListArray_getitem_next_at"
    pattern = namespace["_cuda_error_pattern"]("index out of range", name)
    with pytest.raises(ValueError, match=pattern) as caught:
        checked()
    assert str(caught.value) == NumpyBackend.instance().format_kernel_error(error)
    assert not re.match(pattern, str(caught.value) + " extra text")
    legacy = namespace["_cuda_error_pattern"](
        "bad index", "awkward_IndexedArray_validity"
    )
    assert re.fullmatch(
        legacy, "bad index in compiled CUDA code (awkward_IndexedArray_validity)"
    )


@pytest.mark.parametrize("code", range(5))
def test_prepared_jagged_apply_uses_public_error_format(code, implementation):
    # Exercise the separate prepared-status decoder, not just the fallback
    # _error helper: GPU dispatch uses this decoder for jagged application.
    attempt = -5 if code in (1, 4) else ak._util.kSliceNone
    status = np.array([2 * 5 + code, attempt], dtype=np.int64).view(_DeviceArray)
    error = implementation["_apply_error"](status)
    assert (error.id, error.attempt) == (2, attempt)
    message = NumpyBackend.instance().format_kernel_error(error)
    suffix = " (in compiled code: cuda.compute: awkward_ListArray_getitem_jagged_apply)"
    expected = error.str.decode()
    if attempt != ak._util.kSliceNone:
        expected += f" while attempting to get index {attempt}"
    assert message == expected + suffix


def test_gpu_error_format_regression_uses_canonical_argument_order(implementation):
    # Execute the GPU regression's actual function with host array substitutes.
    # This checks its full call wiring, including interleaved scalar arguments.
    path = ROOT / "tests-cuda/test_4481_listarray_compute.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_prepared_jagged_apply_public_error_format"
    )

    class Backend:
        def __getitem__(self, key):
            assert key == ("awkward_ListArray_getitem_jagged_apply", *([np.int64] * 7))
            return implementation["jagged_apply"]

        def format_kernel_error(self, error):
            return NumpyBackend.instance().format_kernel_error(error)

    namespace = {
        "np": np,
        "cp": np,
        "CupyBackend": SimpleNamespace(instance=Backend),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace[function.name]()


@pytest.mark.parametrize("rows,columns", [(1, 0), (1, 1), (4, 129), (65, 2)])
@pytest.mark.parametrize("bad", [None, 0, 64, 128])
def test_prepared_next_array_chunks_reuse(
    rows, columns, bad, prepared_implementation, implementation
):
    namespace, buffer, events = prepared_implementation
    namespace["_setup"].cache_clear()
    starts = np.arange(rows, dtype=np.int64) * max(columns, 1)
    indices = np.arange(columns, dtype=np.int64)
    if bad is not None and bad < columns:
        indices[bad] = columns
    output = np.full(rows * columns + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        output.copy(),
        starts,
        starts + columns,
        indices,
        rows,
        columns,
        rows * max(columns, 1),
    ]
    reference = copy.deepcopy(args)
    name = "awkward_ListArray_getitem_next_array"
    expected = NumpyBackend.instance()[(name, *([np.int64] * 5))](*reference)
    first_builds = None
    for _ in range(3):
        device = [
            x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args
        ]
        actual = run_prepared(namespace, implementation, name, device)
        assert_error(actual, expected)
        for i in (0, 1):
            np.testing.assert_array_equal(device[i], reference[i])
        builds = sum(event[0] in ("prepare", "prepare_reduce") for event in events)
        if first_builds is None:
            first_builds = builds
        assert builds == first_builds


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_prepared_first_error_rebinds_without_rebuilding(
    dtype, prepared_implementation
):
    namespace, buffer, events = prepared_implementation
    namespace["_setup"].cache_clear()
    events.clear()
    for size, bad in [(1, None), (65, 0), (129, 64), (4097, 4096), (65, 32), (1, 0)]:
        codes = np.zeros(size, dtype=dtype).view(buffer)
        if bad is not None:
            codes[bad] = 3
        actual = namespace["run"]("first_error", (codes,), size, 0, SimpleNamespace())
        assert actual[0] == (size if bad is None else bad)
    assert sum(event[0] == "prepare_reduce" for event in events) == 1
    assert namespace["_setup"].cache_info().misses == 1
