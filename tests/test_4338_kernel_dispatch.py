# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ctypes
import importlib

import awkward_cpp
import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._kernels import CTypesKernel, JaxKernel, NumpyKernel

KEY = ("awkward_ByteMaskedArray_numnull", np.int64, np.int8)
MISSING_KEY = ("awkward_not_a_kernel", np.int64)


def fresh_backend(cls):
    # the backends are singletons, so reach past `instance()` for a cold cache
    backend = object.__new__(cls)
    backend.__init__()
    return backend


@pytest.mark.parametrize("cls", [NumpyBackend, TypeTracerBackend])
def test_fresh_backend_has_empty_kernel_cache(cls):
    backend = fresh_backend(cls)
    assert backend._kernels == {}
    assert backend[KEY] is backend[KEY]
    assert list(backend._kernels) == [KEY]


def test_failed_lookup_is_not_cached():
    backend = fresh_backend(NumpyBackend)
    for _ in range(2):
        with pytest.raises(KeyError):
            backend[MISSING_KEY]
    assert MISSING_KEY not in backend._kernels


@pytest.mark.parametrize("module,path", [("cupy", "cupy"), ("jax", "jax")])
def test_fresh_device_backend_has_empty_kernel_cache(module, path):
    pytest.importorskip(module)
    if module == "jax":
        ak.jax.register_and_check()
    backend_module = importlib.import_module(f"awkward._backends.{path}")
    cls = getattr(backend_module, f"{module.capitalize()}Backend")
    assert fresh_backend(cls)._kernels == {}


def test_ctypes_kernel_pointer_of_is_abstract():
    kernel = CTypesKernel(awkward_cpp.cpu_kernels.kernel[KEY], KEY)
    with pytest.raises(NotImplementedError):
        kernel._pointer_of(np.arange(3))


def test_numpy_pointer_of():
    kernel = NumpyKernel(awkward_cpp.cpu_kernels.kernel[KEY], KEY)

    array = np.arange(3, dtype=np.int64)
    assert kernel._pointer_of(array) == array.ctypes.data

    # a 0-d array is passed through untouched
    scalar = np.array(5, dtype=np.int64)
    assert kernel._pointer_of(scalar) is scalar

    assert isinstance(kernel._pointer_of((ctypes.c_int64 * 3)()), ctypes.c_void_p)

    with pytest.raises(AssertionError, match="Only NumPy buffers"):
        kernel._pointer_of([1, 2, 3])


class FakeCPUOnlyJax:
    def is_own_array(self, x):
        return True

    def is_tracer_type(self, type_):
        return False

    def is_c_contiguous(self, x):
        return True


class FakeGPUArray:
    ndim = 1
    device = type("device", (), {"platform": "gpu"})()


@pytest.fixture(scope="module")
def jax_kernel():
    pytest.importorskip("jax")
    ak.jax.register_and_check()
    return JaxKernel(awkward_cpp.cpu_kernels.kernel[KEY], KEY)


def test_jax_pointer_of_rejects_tracer(jax_kernel):
    jax = pytest.importorskip("jax")

    seen = []
    jax.jit(lambda x: (seen.append(x), x * 2)[1])(jax.numpy.array([1.0, 2.0]))

    with pytest.raises(ValueError, match=KEY[0]):
        jax_kernel._pointer_of(seen[0])


def test_jax_pointer_of_rejects_non_cpu_device(jax_kernel, monkeypatch):
    monkeypatch.setattr(jax_kernel, "_jax", FakeCPUOnlyJax())
    with pytest.raises(RuntimeError, match="requires CPU JAX buffers"):
        jax_kernel._pointer_of(FakeGPUArray())


def test_jax_pointer_of(jax_kernel):
    jax = pytest.importorskip("jax")

    array = jax.numpy.array([1, 2, 3])
    assert isinstance(jax_kernel._pointer_of(array), int)

    # a 0-d array is passed through untouched
    scalar = jax.numpy.array(5)
    assert jax_kernel._pointer_of(scalar) is scalar

    assert isinstance(jax_kernel._pointer_of((ctypes.c_int64 * 3)()), ctypes.c_void_p)

    with pytest.raises(AssertionError, match="Only JAX buffers"):
        jax_kernel._pointer_of([1, 2, 3])


def test_kernels_still_give_the_same_answers():
    array = ak.Array([[1, 2, 3], [], [4, 5]])

    assert ak.num(array).to_list() == [3, 0, 2]
    assert ak.flatten(array).to_list() == [1, 2, 3, 4, 5]
    assert array[:, 1:].to_list() == [[2, 3], [], [5]]
    assert ak.to_list(array[[0, 2]]) == [[1, 2, 3], [4, 5]]

    typetracer = array.layout.to_typetracer(forget_length=True)
    assert str(ak.num(ak.Array(typetracer)).type) == "## * int64"


def test_jax_pointer_of_rejects_autodiff_tracer(jax_kernel):
    jax = pytest.importorskip("jax")

    def f(x):
        with pytest.raises(ValueError, match="not differentiable"):
            jax_kernel._pointer_of(x)
        return x

    one = jax.numpy.array([1.0])
    jax.jvp(f, (one,), (one,))
