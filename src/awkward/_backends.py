from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TypeVar

import awkward_cpp
from typing_extensions import Protocol, TypeAlias, Unpack, runtime_checkable

import awkward as ak
from awkward._typetracer import NoKernel, TypeTracer
from awkward.nplikes import (
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

np = NumpyMetadata.instance()


T = TypeVar("T")
KernelKeyType: TypeAlias = "tuple[str, Unpack[tuple[np.dtype, ...]]]"
KernelType: TypeAlias = "Callable[[...], None]"


@runtime_checkable
class Backend(Protocol[T]):
    @property
    @abstractmethod
    def nplike(self) -> NumpyLike:
        ...

    @property
    @abstractmethod
    def index_nplike(self) -> NumpyLike:
        ...

    def __getitem__(self, key: KernelKeyType) -> KernelType:
        ...


class NumpyBackend(Backend[Any], Singleton):
    _numpy: Numpy

    @property
    def nplike(self) -> Numpy:
        return self._numpy

    @property
    def index_nplike(self) -> Numpy:
        return self._numpy

    def __init__(self):
        self._numpy = Numpy.instance()

    def __getitem__(self, index: KernelKeyType) -> KernelType[Any]:
        return NumpyKernel(awkward_cpp.cpu_kernels.kernel[index], index)


class CupyBackend(Backend[Any], Singleton):
    _cupy: Cupy

    @property
    def nplike(self) -> Cupy:
        return self._cupy

    @property
    def index_nplike(self) -> Cupy:
        return self._cupy

    def __init__(self):
        self._cupy = Cupy.instance()

    def __getitem__(self, index: KernelKeyType) -> KernelType[Any]:
        import awkward._connect.cuda as cuda

        cupy = cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = cuda.initialize_cuda_kernels(cupy)
        func = _cuda_kernels[index]
        if func is not None:
            return CupyKernel(func, index)
        return NumpyKernel(awkward_cpp.cpu_kernels.kernel[index], index)


class JaxBackend(Backend[Any], Singleton):
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

    def __getitem__(self, index: KernelKeyType) -> KernelType[Any]:
        # JAX uses Awkward's C++ kernels for index-only operations
        return JaxKernel(awkward_cpp.cpu_kernels.kernel[index], index)


class TypeTracerBackend(Backend[Any], Singleton):
    _typetracer: TypeTracer

    @property
    def nplike(self) -> TypeTracer:
        return self._typetracer

    @property
    def index_nplike(self) -> TypeTracer:
        return self._typetracer

    def __init__(self):
        self._typetracer = TypeTracer.instance()

    def __getitem__(self, index: KernelKeyType) -> KernelType[Any]:
        return NoKernel(index)


_UNSET = object()
D = TypeVar("D")


def _backend_for_nplike(nplike: ak.nplikes.NumpyLike) -> Backend:
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


def backend_for(*arrays, default: T = _UNSET) -> Backend | D:
    nplike = nplike_of(*arrays, default=None)
    if nplike is None:
        if default is _UNSET:
            raise ak._errors.wrap_error(
                ValueError("could not find backend for", arrays)
            )
        else:
            return default
    else:
        return _backend_for_nplike(nplike)
