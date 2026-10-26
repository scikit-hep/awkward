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
FUNCTIONS = {
    "awkward_ListOffsetArray_flatten_offsets": "flatten_offsets",
    "awkward_ListOffsetArray_toRegularArray": "to_regular",
    "awkward_ListOffsetArray_rpad_axis1": "rpad",
    "awkward_ListOffsetArray_drop_none_indexes": "drop_none",
    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64": "nonlocal_ranges",
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


class _Discard:
    pass


def host_implementation():
    def array_factory(function):
        def allocate(*args, **kwargs):
            return function(*args, **kwargs).view(_DeviceArray)

        return allocate

    cp = SimpleNamespace(
        **{
            name: array_factory(getattr(np, name))
            for name in ("empty", "zeros", "full", "asarray", "arange")
        },
        int64=np.int64,
        maximum=np.maximum,
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
        operation = kwargs["op"]
        value = int(h_init)
        for item in d_in[:num_items]:
            value = operation(value, item)
        d_out[0] = value

    def segmented(
        *, d_in, d_out, num_segments, start_offsets_in, end_offsets_in, h_init, **kwargs
    ):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        # Exercise the supported array-input path, not a stateful iterator.
        assert isinstance(d_in, np.ndarray)
        assert d_in.dtype == np.dtype(np.int64)
        assert kwargs["max_segment_size"] is not None
        for row in range(num_segments):
            values = [
                d_in[i] for i in range(start_offsets_in[row], end_offsets_in[row])
            ]
            d_out[row] = max([int(h_init), *values])

    tree = ast.parse(
        (ROOT / "src/awkward/_connect/cuda/_listoffsetarray.py").read_text()
    )
    body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {
        "__name__": __name__,
        "np": np,
        "cp": cp,
        "dataclass": dataclass,
        "_SLICE_NONE": ak._util.kSliceNone,
        "CountingIterator": _Counting,
        "segmented_reduce": segmented,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(PLUS=0, MINIMUM=1, MAXIMUM=2),
        "unary_transform": transform,
        "inclusive_scan": scan,
        "reduce_into": reduce,
    }
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(ROOT / "src/awkward/_connect/cuda/_listoffsetarray.py"),
            "exec",
        ),
        namespace,
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
    if name not in {
        "awkward_ListOffsetArray_toRegularArray",
        "awkward_ListOffsetArray_drop_none_indexes",
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


@pytest.mark.parametrize("offsets", [[2, 4, 6], [0, 0, 0], [0], [0, 2, 5]])
def test_public_regular_conversion(compute_backend, offsets):
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.asarray(offsets)),
        ak.contents.NumpyArray(np.arange(6)),
    )
    expected = ak.Array(layout).to_list()
    layout._backend = compute_backend
    if offsets == [0, 2, 5]:
        with pytest.raises(ValueError, match="not regular"):
            ak.to_regular(layout)
    else:
        assert ak.to_regular(layout).to_list() == expected


@pytest.mark.parametrize("groups,width", [(1, 1), (17, 33), (257, 513)])
def test_nonlocal_ranges_across_block_boundaries(implementation, groups, width):
    rng = np.random.default_rng(2026)
    distincts = rng.choice([-1, -2, 0, 7], groups * width).astype(np.int64)
    starts = np.full(groups + 3, 123, dtype=np.int64)
    stops = starts.copy()
    expected_starts, expected_stops = starts.copy(), stops.copy()
    kernel = NumpyBackend.instance()[
        "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
        np.int64,
        np.int64,
        np.int64,
    ]
    kernel(expected_starts, expected_stops, distincts, distincts.size, groups)
    implementation["nonlocal_ranges"](starts, stops, distincts, distincts.size, groups)
    np.testing.assert_array_equal(starts, expected_starts)
    np.testing.assert_array_equal(stops, expected_stops)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("offsets", [[0], [2, 4, 4, 7]])
@pytest.mark.parametrize("target", [0, 2, 4])
def test_pad_none_offset_widths(dtype, offsets, target):
    source = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index(np.asarray(offsets, dtype=dtype)),
            ak.contents.NumpyArray(np.arange(9)),
        )
    )
    padded = ak.pad_none(source, target)
    expected = [row + [None] * max(0, target - len(row)) for row in source.to_list()]
    assert padded.to_list() == expected
    assert padded.layout.offsets.dtype == np.dtype(np.int64)
    assert (
        ak.pad_none(
            source.layout.to_typetracer(forget_length=True), target, highlevel=False
        ).form
        == padded.layout.form
    )


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("rows", [0, 3])
def test_padding_length_scan_uses_current_stream(dtype, rows):
    source = ROOT / "src/awkward/_connect/cuda/_compute.py"
    node = next(
        x
        for x in ast.parse(source.read_text()).body
        if isinstance(x, ast.FunctionDef)
        and x.name == "awkward_ListOffsetArray_rpad_length_axis1"
    )
    node.body = [x for x in node.body if not isinstance(x, ast.ImportFrom)]
    namespace = host_implementation()
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    offsets = np.arange(rows + 1, dtype=dtype) * 2
    output = np.full(rows + 3, 99, dtype=dtype)
    length = np.full(3, 99, dtype=np.int64)
    namespace[node.name](output, offsets, rows, 4, length)
    assert output.tolist() == [4 * i for i in range(rows + 1)] + [99, 99]
    assert length.tolist() == [4 * rows, 99, 99]


def test_prepared_offset_transforms_reuse_setup_without_retaining_buffers():
    import gc
    import weakref
    from functools import lru_cache
    from threading import RLock

    registry = weakref.WeakValueDictionary()
    builds = []

    class Buffer(np.ndarray):
        @property
        def __cuda_array_interface__(self):
            pointer = self.ctypes.data
            registry[pointer] = self
            return {
                "shape": self.shape,
                "strides": None,
                "typestr": self.dtype.str,
                "data": (pointer, False),
                "version": 3,
            }

    def prepare(**kwargs):
        builds.append(True)
        operation = kwargs["op"]

        def launch(**kwargs):
            import time

            time.sleep(0.001)
            for i in range(kwargs["num_items"]):
                value = operation(i)
                if not isinstance(kwargs["d_out"], _Discard):
                    kwargs["d_out"][i] = value

        return launch

    def prepare_reduce(**kwargs):
        builds.append(True)

        def launch(*, temp_storage, **kwargs):
            if temp_storage is None:
                return 64
            value = int(kwargs["h_init"])
            for item in kwargs["d_in"][: kwargs["num_items"]]:
                value = kwargs["op"](value, item)
            kwargs["d_out"][0] = value
            return 64

        return launch

    def prepare_scan(**kwargs):
        builds.append(True)

        def launch(*, temp_storage, **kwargs):
            if temp_storage is None:
                return 64
            length = kwargs["num_items"]
            kwargs["d_out"][:length] = np.cumsum(kwargs["d_in"][:length])
            return 64

        return launch

    def prepare_segmented(**kwargs):
        builds.append(True)

        def launch(*, temp_storage, **kwargs):
            if temp_storage is None:
                return 64
            for i in range(kwargs["num_segments"]):
                start = kwargs["start_offsets_in"][i]
                stop = kwargs["end_offsets_in"][i]
                kwargs["d_out"][i] = max(kwargs["d_in"][start:stop])
            return 64

        return launch

    namespace = {
        "np": np,
        "cp": SimpleNamespace(
            empty=lambda *args, **kwargs: np.empty(*args, **kwargs).view(Buffer),
            int64=np.int64,
            uint8=np.uint8,
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDevice=lambda: 0),
                driver=SimpleNamespace(ctxGetCurrent=lambda: 1),
            ),
        ),
        "CountingIterator": _Counting,
        "make_op_adapter": lambda op: op,
        "make_unary_transform": prepare,
        "make_reduce_into": prepare_reduce,
        "make_segmented_reduce": prepare_segmented,
        "make_inclusive_scan": prepare_scan,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(MAXIMUM=2, PLUS=0),
        "lru_cache": lru_cache,
        "_lock": RLock(),
    }
    path = ROOT / "src/awkward/_connect/cuda/_prepared_gather.py"
    tree = ast.parse(path.read_text())
    exec(
        compile(
            ast.Module(
                body=[
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                ],
                type_ignores=[],
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    state = namespace["_ArrayState"]
    state.__getitem__ = lambda self, i: registry[self.data_ptr()][i]
    gather = namespace["gather_offsets"]
    for length in (3, 2, 4):
        outer = np.arange(length, dtype=np.int32).view(Buffer)
        inner = (np.arange(length, dtype=np.int64) + 10 * length).view(Buffer)
        output = np.full(length + 2, -1, dtype=np.int64).view(Buffer)
        references = [weakref.ref(x) for x in (outer, inner, output)]
        gather(output, outer, inner, length, None)
        np.testing.assert_array_equal(output[:length], inner)
        np.testing.assert_array_equal(output[length:], [-1, -1])
        setup = namespace["_setup"](
            0, 1, outer.dtype.str, inner.dtype.str, output.dtype.str
        )
        assert setup.outer.data_ptr() == setup.inner.data_ptr() == 0
        del outer, inner, output
        gc.collect()
        assert all(reference() is None for reference in references)
    assert len(builds) == 1

    from concurrent.futures import ThreadPoolExecutor

    def concurrent_launch(value):
        outer = np.arange(3, dtype=np.int32).view(Buffer)
        inner = np.full(3, value, dtype=np.int64).view(Buffer)
        output = np.zeros(3, dtype=np.int64).view(Buffer)
        gather(output, outer, inner, 3, None)
        np.testing.assert_array_equal(output, [value] * 3)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(concurrent_launch, range(12)))
    assert len(builds) == 1

    def fail(**kwargs):
        raise RuntimeError("launch failed")

    setup.prepared = fail
    outer = np.arange(2, dtype=np.int32).view(Buffer)
    inner = np.arange(2, dtype=np.int64).view(Buffer)
    output = np.zeros(2, dtype=np.int64).view(Buffer)
    with pytest.raises(RuntimeError, match="launch failed"):
        gather(output, outer, inner, 2, None)
    assert setup.outer.data_ptr() == setup.inner.data_ptr() == 0

    state.__setitem__ = lambda self, i, value: registry[self.data_ptr()].__setitem__(
        i, value
    )
    ranges = namespace["nonlocal_offsets"]
    initial_builds = len(builds)
    for width, groups in (
        (1, 3),
        (8, 3),
        (64, 3),
        (65, 3),
        (4096, 5),
        (8, 5),
        (4096, 5),
    ):
        source = np.full(groups * width, -1, dtype=np.int64).view(Buffer)
        source[width] = -2
        source[3 * width - 1] = 7
        starts = np.full(groups + 3, 123, dtype=np.int64).view(Buffer)
        stops = starts.copy().view(Buffer)
        expected_starts, expected_stops = starts.copy(), stops.copy()
        NumpyBackend.instance()[
            "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
            np.int64,
            np.int64,
            np.int64,
        ](expected_starts, expected_stops, source, source.size, groups)
        references = [weakref.ref(x) for x in (source, starts, stops)]
        ranges(starts, stops, source, source.size, groups, None)
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(stops, expected_stops)
        range_setup = namespace["_ranges_setup"](
            0,
            1,
            starts.dtype.str,
            source.dtype.str,
            stops.dtype.str,
            width,
            groups if width > 64 else 0,
        )
        assert range_setup.outer.data_ptr() == range_setup.inner.data_ptr() == 0
        if width > 64:
            assert range_setup.boundaries.data_ptr() == 0
        del source, starts, stops
        gc.collect()
        assert all(reference() is None for reference in references)
    assert len(builds) - initial_builds == 7

    def concurrent_ranges(value):
        source = np.full(3 * 65, value, dtype=np.int64).view(Buffer)
        starts = np.full(5, 123, dtype=np.int64).view(Buffer)
        stops = starts.copy().view(Buffer)
        ranges(starts, stops, source, source.size, 3, None)
        np.testing.assert_array_equal(starts, [0, 65, 130, 123, 123])
        expected = [0, 65, 130] if value == -1 else [65, 130, 195]
        np.testing.assert_array_equal(stops, [*expected, 123, 123])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(concurrent_ranges, [-1, -2, 0, 7] * 3))
    assert len(builds) - initial_builds == 7

    initial_builds = len(builds)
    for dtype in (np.int32, np.uint32, np.int64):
        for values, code, expected_size in (
            ([5, 7, 9], 0, 2),
            ([0, 2, 5, 1], 2, 2),
            ([5, 3, 4], -1, -1),
            ([0, 2, 1, 9], -2, 2),
            ([0, 0, 0, 0, 0], 0, 0),
        ):
            offsets = np.asarray(values, dtype=dtype).view(Buffer)
            size = np.full(3, 123, dtype=np.int64).view(Buffer)
            references = [weakref.ref(x) for x in (offsets, size)]
            status = namespace["regular_status"](size, offsets, len(offsets), None)
            assert status.shape == (1,)
            assert status.nbytes == 8
            assert int(status[0]) == code
            np.testing.assert_array_equal(size, [expected_size, 123, 123])
            del offsets, size, status
            gc.collect()
            assert all(reference() is None for reference in references)
    assert len(builds) - initial_builds == 6

    seen_drop_setups = set()
    for dtype in (np.int32, np.int64):
        cases = [
            ([0, 2, 2, 5], [-1, 0, -2, 2, -1], 5),
            ([2, 4, 5], [-1, 0, -2, 2, -1], 5),
            ([3, 1, 5], [-1, 0, -2, 2, -1], 5),
            ([0, 6], [-1, 0, -2, 2, -1], 5),
            ([0, 0, 0], [], 0),
            ([1], [-2, -1], 2),
            ([0, 4096, 8192], [-1, 0] * 4096, 8192),
        ]
        for values, entries, declared_length in cases * 2:
            offsets = np.asarray(values, dtype=dtype).view(Buffer)
            indexes = np.asarray(entries, dtype=dtype).view(Buffer)
            output = np.full(len(values) + 3, 123, dtype=dtype).view(Buffer)
            expected = output.copy()
            error = NumpyBackend.instance()[
                "awkward_ListOffsetArray_drop_none_indexes", dtype, dtype, dtype
            ](expected, indexes, offsets, len(values), declared_length)
            references = [weakref.ref(x) for x in (offsets, indexes, output)]
            before = len(builds)
            status = namespace["drop_status"](
                output, indexes, offsets, len(values), declared_length, None
            )
            assert (int(status[0]) < 0) == bool(error.str)
            np.testing.assert_array_equal(output, expected)
            # Each variant compiles its count/scan/write only on first use.
            key = (np.dtype(dtype).str, len(values), declared_length)
            assert len(builds) - before == (0 if key in seen_drop_setups else 3)
            seen_drop_setups.add(key)
            del offsets, indexes, output, status
            gc.collect()
            assert all(reference() is None for reference in references)

    before = len(builds)

    def concurrent_drop(value):
        offsets = np.asarray([0, 2, 3, 5], dtype=np.int64).view(Buffer)
        indexes = np.full(5, value, dtype=np.int64).view(Buffer)
        output = np.full(6, 123, dtype=np.int64).view(Buffer)
        status = namespace["drop_status"](output, indexes, offsets, 4, 5, None)
        assert int(status[0]) >= 0
        expected = [0] * 4 if value < 0 else [0, 2, 3, 5]
        np.testing.assert_array_equal(output, [*expected, 123, 123])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(concurrent_drop, [-1, -2, 0, 7] * 3))
    assert len(builds) == before
