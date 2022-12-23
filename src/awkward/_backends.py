from __future__ import annotations

from abc import ABC, abstractmethod

import awkward_cpp

import awkward as ak
from awkward._nplikes import (
    Cupy,
    CupyKernel,
    Jax,
    JaxKernel,
    Numpy,
    NumpyKernel,
    NumpyLike,
    NumpyMetadata,
    Singleton,
    nplike_of,
)
from awkward._typetracer import NoKernel, TypeTracer
from awkward.typing import Callable, Final, Tuple, TypeAlias, TypeVar, Unpack

np = NumpyMetadata.instance()


T = TypeVar("T", covariant=True)
KernelKeyType: TypeAlias = Tuple[str, Unpack[Tuple[np.dtype, ...]]]
KernelType: TypeAlias = Callable[..., None]


class Backend(Singleton, ABC):
    name: str

    @property
    @abstractmethod
    def nplike(self) -> NumpyLike:
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    @abstractmethod
    def index_nplike(self) -> NumpyLike:
        raise ak._errors.wrap_error(NotImplementedError)

    def __getitem__(self, key: KernelKeyType) -> KernelType:
        raise ak._errors.wrap_error(NotImplementedError)

    def apply_reducer(
        self,
        reducer: ak._reducers.Reducer,
        layout: ak.contents.NumpyArray,
        parents: ak.index.Index,
        outlength: int,
    ) -> ak.contents.NumpyArray:
        return reducer.apply(layout, parents, outlength)

    def apply_ufunc(self, ufunc, method, args, kwargs):
        return getattr(ufunc, method)(*args, **kwargs)


class NumpyBackend(Backend):
    name: Final[str] = "cpu"

    _numpy: Numpy

    @property
    def nplike(self) -> Numpy:
        return self._numpy

    @property
    def index_nplike(self) -> Numpy:
        return self._numpy

    def __init__(self):
        self._numpy = Numpy.instance()

    def __getitem__(self, index: KernelKeyType) -> NumpyKernel:
        return NumpyKernel(awkward_cpp.cpu_kernels.kernel[index], index)


class CupyBackend(Backend):
    name: Final[str] = "cuda"

    _cupy: Cupy

    @property
    def nplike(self) -> Cupy:
        return self._cupy

    @property
    def index_nplike(self) -> Cupy:
        return self._cupy

    def __init__(self):
        self._cupy = Cupy.instance()

    def __getitem__(self, index: KernelKeyType) -> CupyKernel | NumpyKernel:
        from awkward._connect import cuda

        cupy = cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = cuda.initialize_cuda_kernels(cupy)
        func = _cuda_kernels[index]
        if func is not None:
            return CupyKernel(func, index)
        else:
            raise ak._errors.wrap_error(
                AssertionError(f"CuPyKernel not found: {index!r}")
            )


class JaxBackend(Backend):
    name: Final[str] = "jax"

    _jax: Jax
    _numpy: Numpy

    @property
    def nplike(self) -> Jax:
        return self._jax

    @property
    def index_nplike(self) -> Numpy:
        return self._numpy

    def __init__(self):
        self._jax = Jax.instance()
        self._numpy = Numpy.instance()

    def __getitem__(self, index: KernelKeyType) -> JaxKernel:
        # JAX uses Awkward's C++ kernels for index-only operations
        return JaxKernel(awkward_cpp.cpu_kernels.kernel[index], index)

    def apply_reducer(
        self,
        reducer: ak._reducers.Reducer,
        layout: ak.contents.NumpyArray,
        parents: ak.index.Index,
        outlength: int,
    ) -> ak.contents.NumpyArray:
        from awkward._connect.jax import get_jax_reducer

        jax_reducer = get_jax_reducer(reducer)
        return jax_reducer.apply(layout, parents, outlength)

    def apply_ufunc(self, ufunc, method, args, kwargs):
        from awkward._connect.jax import get_jax_ufunc

        if method != "__call__":
            raise ak._errors.wrap_error(
                ValueError(f"unsupported ufunc method {method} called")
            )

        jax_ufunc = get_jax_ufunc(ufunc)
        return jax_ufunc(*args, **kwargs)


class TypeTracerBackend(Backend):
    name: Final[str] = "typetracer"

    _typetracer: TypeTracer

    @property
    def nplike(self) -> TypeTracer:
        return self._typetracer

    @property
    def index_nplike(self) -> TypeTracer:
        return self._typetracer

    def __init__(self):
        self._typetracer = TypeTracer.instance()

    def __getitem__(self, index: KernelKeyType) -> NoKernel:
        return NoKernel(index)


def _backend_for_nplike(nplike: ak._nplikes.NumpyLike) -> Backend:
    # Currently there exists a one-to-one relationship between the nplike
    # and the backend. In future, this might need refactoring
    if isinstance(nplike, Numpy):
        return NumpyBackend.instance()
    elif isinstance(nplike, Cupy):
        return CupyBackend.instance()
    elif isinstance(nplike, Jax):
        return JaxBackend.instance()
    elif isinstance(nplike, TypeTracer):
        return TypeTracerBackend.instance()
    else:
        raise ak._errors.wrap_error(ValueError("unrecognised nplike", nplike))


_UNSET = object()
D = TypeVar("D")


def backend_of(*objects, default: D = _UNSET) -> Backend | D:
    """
    Args:
        objects: objects for which to find a suitable backend
        default: value to return if no backend is found.

    Return the most suitable backend for the given objects (e.g. arrays, layouts). If no
    suitable backend is found, return the `default` value, or raise a `ValueError` if
    no default is given.
    """
    nplike = nplike_of(*objects, default=None)
    if nplike is not None:
        return _backend_for_nplike(nplike)
    elif default is _UNSET:
        raise ak._errors.wrap_error(ValueError("could not find backend for", objects))
    else:
        return default


_backends: Final[dict[str, type[Backend]]] = {
    b.name: b for b in (NumpyBackend, CupyBackend, JaxBackend, TypeTracerBackend)
}


def regularize_backend(backend: str | Backend) -> Backend:
    if isinstance(backend, Backend):
        return backend
    elif backend in _backends:
        return _backends[backend].instance()
    else:
        raise ak._errors.wrap_error(ValueError(f"No such backend {backend!r} exists."))
