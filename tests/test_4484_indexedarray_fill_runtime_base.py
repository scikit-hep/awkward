# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Import the helper with host CUDA doubles; device compilation is in tests-cuda."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Dependency replacement and prepared caches are process-wide test state.
pytestmark = pytest.mark.thread_unsafe


@pytest.fixture
def implementation(monkeypatch):
    from awkward._nplikes.cupy import Cupy

    registry, events = {}, []

    from threading import local

    class DefaultStream:
        def __init__(self):
            self.state = local()

        @property
        def active(self):
            return getattr(self.state, "active", False)

        def __enter__(self):
            self.state.active = True

        def __exit__(self, *args):
            self.state.active = False

    stream = DefaultStream()

    class Array(np.ndarray):
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

    def asarray(value, dtype):
        assert stream.active, "parameter upload must use the legacy stream"
        return np.asarray(value, dtype=dtype).view(Array)

    def prepare(**options):
        events.append("prepare")

        def launch(**kw):
            assert kw["stream"] is None
            assert stream.active
            for i in range(kw["num_items"]):
                kw["d_out"][i] = kw["op"](kw["d_in"][i])

        return launch

    cp = SimpleNamespace(
        asarray=asarray,
        int64=np.int64,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: 0),
            driver=SimpleNamespace(ctxGetCurrent=lambda: 42),
            Stream=SimpleNamespace(null=stream),
        ),
    )
    cache = ModuleType("awkward._connect.cuda._compute_cache")
    cache.load_or_build = lambda factory, **kwargs: factory(
        **{key: value for key, value in kwargs.items() if key != "source_path"}
    )
    monkeypatch.setitem(sys.modules, "awkward._connect.cuda._compute_cache", cache)
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
    source = (
        Path(__file__).resolve().parents[1]
        / "src/awkward/_connect/cuda/_indexed_fill.py"
    )
    spec = importlib.util.spec_from_file_location("indexed_fill_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = vars(module)
    namespace["_BaseState"].__getitem__ = lambda self, i: registry[
        self.__cuda_array_interface__["data"][0]
    ][i]
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


# These tests inspect or replace cached setup shared by the fixture threads.
@pytest.mark.thread_unsafe
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
    assert setup.base.__cuda_array_interface__["data"][0] == 0
    assert namespace["_setup"].cache_info().maxsize == 32


@pytest.mark.thread_unsafe
def test_descriptor_is_cleared_when_launch_fails(implementation):
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
    assert setup.base.__cuda_array_interface__["data"][0] == 0


def test_descriptor_is_cleared_when_preparation_fails(implementation):
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
    assert setup.base.__cuda_array_interface__["data"][0] == 0


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
