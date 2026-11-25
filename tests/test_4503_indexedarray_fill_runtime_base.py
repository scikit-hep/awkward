# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Import the helper with host CUDA doubles; device compilation is in tests-cuda."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import awkward as ak

# Dependency replacement and prepared caches are process-wide test state.
pytestmark = pytest.mark.thread_unsafe


@pytest.fixture
def implementation(monkeypatch):
    from awkward._nplikes.cupy import Cupy

    events = []

    from threading import local

    class DefaultStream:
        def __init__(self):
            self.state = local()
            self.defer = False
            self.queue = []

        @property
        def active(self):
            return getattr(self.state, "active", False)

        def __enter__(self):
            self.state.active = True

        def __exit__(self, *args):
            self.state.active = False

    stream = DefaultStream()

    def submit(operation):
        if stream.defer:
            stream.queue.append(operation)
        else:
            operation()

    class Array(np.ndarray):
        def fill(self, value):
            assert stream.active, "parameter updates must use the legacy stream"
            events.append(("update", int(value)))
            submit(lambda: np.ndarray.fill(self, value))

    def empty(size, dtype):
        assert stream.active, "parameter allocation must use the legacy stream"
        events.append("allocate")
        return np.empty(size, dtype=dtype).view(Array)

    def prepare(**options):
        events.append("prepare")

        def launch(**kw):
            assert kw["stream"] is None
            assert stream.active
            events.append("launch")

            def apply():
                for i in range(kw["num_items"]):
                    kw["d_out"][i] = kw["op"](kw["d_in"][i])

            submit(apply)

        return launch

    cp = SimpleNamespace(
        empty=empty,
        int64=np.int64,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: 0),
            driver=SimpleNamespace(ctxGetCurrent=lambda: 42),
            Stream=SimpleNamespace(null=stream),
        ),
    )
    cuda = ModuleType("cuda")
    cuda.__path__ = []
    compute = ModuleType("cuda.compute")
    compute.make_unary_transform = prepare
    operation = ModuleType("cuda.compute.op")
    operation.make_op_adapter = lambda op: op
    for name, module in (
        ("cuda", cuda),
        ("cuda.compute", compute),
        ("cuda.compute.op", operation),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(Cupy, "instance", lambda: SimpleNamespace(_module=cp))
    source = Path(ak.__file__).parent / "_connect/cuda/_indexed_fill.py"
    spec = importlib.util.spec_from_file_location(
        "awkward._connect.cuda._indexed_fill", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = vars(module)
    namespace["test_stream"] = stream
    return namespace, Array, events


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
@pytest.mark.parametrize("length", [0, 1, 2, 31, 64, 129])
@pytest.mark.parametrize("base", [-10, 0, 10, np.int64(20)])
def test_runtime_base_matches_reference(dtype, length, base, implementation):
    namespace, array, _ = implementation
    data = [-1, 0, 2, -3, 5] if dtype != np.uint32 else [0, 1, 2, 3, 5]
    source = np.resize(np.array(data, dtype=dtype), max(1, length)).view(array)
    output = np.full(length + 9, 123, dtype=np.int64).view(array)
    expected = output.copy()
    for i in range(length):
        x = int(source[i])
        expected[3 + i] = -1 if x < 0 else x + int(base)
    namespace["fill"](output, 3, source, length, base)
    np.testing.assert_array_equal(output, expected)


def test_reuses_operation_across_base_pointer_length_and_scalar_type(implementation):
    namespace, array, events = implementation
    for length, base in [
        (64, 10),
        (64, 11),
        (32, np.int64(12)),
        (129, -3),
        (1, 0),
        (64, 10),
    ]:
        source = np.arange(length, dtype=np.int64).view(array)
        output = np.full(length + 4, 123, dtype=np.int64).view(array)
        namespace["fill"](output, 2, source, length, base)
        np.testing.assert_array_equal(
            output[2 : 2 + length], np.arange(length) + int(base)
        )
        assert events.count("prepare") == 1
    setup = namespace["_setup"](0, 42, np.dtype(np.int64).str, np.dtype(np.int64).str)
    assert setup.base.shape == (1,)
    assert namespace["_setup"].cache_info().maxsize == 32


def test_parameter_is_invalidated_when_launch_fails(implementation):
    namespace, array, _ = implementation
    source = np.arange(2, dtype=np.int64).view(array)
    output = np.zeros(2, dtype=np.int64).view(array)
    namespace["fill"](output, 0, source, 2, 10)
    setup = namespace["_setup"](0, 42, source.dtype.str, output.dtype.str)

    def fail(**kwargs):
        raise RuntimeError("launch failed")

    setup.transform = fail
    with pytest.raises(RuntimeError, match="launch failed"):
        namespace["fill"](output, 0, source, 2, 11)
    assert setup.last_base is None


def test_parameter_is_invalidated_when_preparation_fails(implementation):
    namespace, array, _ = implementation
    source = np.arange(2, dtype=np.int64).view(array)
    output = np.zeros(2, dtype=np.int64).view(array)

    def fail(**kwargs):
        raise RuntimeError("build failed")

    namespace["make_unary_transform"] = fail
    with pytest.raises(RuntimeError, match="build failed"):
        namespace["fill"](output, 0, source, 2, 7)
    setup = namespace["_setup"](0, 42, source.dtype.str, output.dtype.str)
    assert setup.transform is None
    assert setup.last_base is None


def test_unrelated_prepared_objects_can_build_concurrently(implementation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    namespace, array, _ = implementation
    prepare = namespace["make_unary_transform"]
    barrier = Barrier(2)

    def concurrent_prepare(**kwargs):
        barrier.wait(timeout=5)
        return prepare(**kwargs)

    namespace["make_unary_transform"] = concurrent_prepare

    def run(dtype):
        source = np.arange(3, dtype=dtype).view(array)
        output = np.zeros(3, dtype=np.int64).view(array)
        namespace["fill"](output, 0, source, 3, 10)
        np.testing.assert_array_equal(output, [10, 11, 12])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [np.int32, np.int64]))


def test_one_parameter_allocation_and_no_updates_for_repeated_base(implementation):
    namespace, array, events = implementation
    source = np.arange(4, dtype=np.int64).view(array)
    output = np.zeros(4, dtype=np.int64).view(array)
    for base in (7, np.int64(7), 7, -3, -3, 0, 0, 7):
        namespace["fill"](output, 0, source, 4, base)
        np.testing.assert_array_equal(output, np.arange(4) + int(base))
    assert events.count("allocate") == 1
    assert events.count("prepare") == 1
    assert [x for x in events if isinstance(x, tuple)] == [
        ("update", 7),
        ("update", -3),
        ("update", 0),
        ("update", 7),
    ]


def test_zero_length_does_not_allocate_or_update(implementation):
    namespace, array, events = implementation
    source = np.zeros(1, dtype=np.int64).view(array)
    output = np.full(1, 99, dtype=np.int64).view(array)
    namespace["fill"](output, 0, source, 0, 7)
    assert not events
    np.testing.assert_array_equal(output, [99])


def test_queued_calls_read_their_own_base(implementation):
    namespace, array, events = implementation
    stream = namespace["test_stream"]
    stream.defer = True
    source = np.asarray([-1, 0, 2, 3], dtype=np.int64).view(array)
    outputs = []
    for base in (7, 7, -3, 0, 11, 11):
        output = np.full(6, 99, dtype=np.int64).view(array)
        namespace["fill"](output, 1, source, 4, base)
        outputs.append((base, output))
    assert events.count("allocate") == 1
    assert events.count("prepare") == 1
    for _, output in outputs:
        np.testing.assert_array_equal(output, np.full(6, 99))
    for operation in stream.queue:
        operation()
    for base, output in outputs:
        np.testing.assert_array_equal(output, [99, -1, base, base + 2, base + 3, 99])


def test_concurrent_cold_calls_share_ordered_parameter(implementation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    namespace, array, events = implementation
    stream = namespace["test_stream"]
    stream.defer = True
    barrier = Barrier(4)

    def run(base):
        source = np.arange(3, dtype=np.int64).view(array)
        output = np.full(3, 99, dtype=np.int64).view(array)
        barrier.wait(timeout=5)
        namespace["fill"](output, 0, source, 3, base)
        return base, output

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, [7, -3, 0, 11]))
    assert events.count("allocate") == 1
    assert events.count("prepare") == 1
    for operation in stream.queue:
        operation()
    for base, output in outputs:
        np.testing.assert_array_equal(output, np.arange(3) + base)


def test_retry_after_launch_failure_refreshes_parameter(implementation):
    namespace, array, events = implementation
    source = np.arange(3, dtype=np.int64).view(array)
    output = np.zeros(3, dtype=np.int64).view(array)
    namespace["fill"](output, 0, source, 3, 7)
    setup = namespace["_setup"](0, 42, source.dtype.str, output.dtype.str)
    original = setup.transform

    def fail(**kwargs):
        raise RuntimeError("launch failed")

    setup.transform = fail
    with pytest.raises(RuntimeError, match="launch failed"):
        namespace["fill"](output, 0, source, 3, 7)
    setup.transform = original
    namespace["fill"](output, 0, source, 3, 7)
    np.testing.assert_array_equal(output, [7, 8, 9])
    assert events.count(("update", 7)) == 2
    assert events.count("allocate") == 1


def test_failed_parameter_update_can_be_retried(implementation, monkeypatch):
    namespace, array, events = implementation
    source = np.arange(3, dtype=np.int64).view(array)
    output = np.full(3, 99, dtype=np.int64).view(array)

    def fail(self, value):
        raise RuntimeError("parameter update failed")

    with monkeypatch.context() as patch:
        patch.setattr(array, "fill", fail)
        with pytest.raises(RuntimeError, match="parameter update failed"):
            namespace["fill"](output, 0, source, 3, 7)
    np.testing.assert_array_equal(output, [99, 99, 99])
    setup = namespace["_setup"](0, 42, source.dtype.str, output.dtype.str)
    assert setup.last_base is None
    assert setup.transform is None
    namespace["fill"](output, 0, source, 3, 7)
    np.testing.assert_array_equal(output, [7, 8, 9])
    assert events.count("allocate") == 1
