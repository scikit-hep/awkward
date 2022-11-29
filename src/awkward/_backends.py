from abc import abstractmethod
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

import awkward_cpp
from typing_extensions import Unpack

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
)

np = NumpyMetadata.instance()


T = TypeVar("T")
KernelKeyType = tuple[str, Unpack[tuple[np.dtype, ...]]]
KernelType = Callable[[tuple[T, ...]], None]


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


class NumpyBackend(Backend[Any]):
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


class CupyBackend(Backend[Any]):
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


class JaxBackend(Backend[Any]):
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


class TypeTracerBackend(Backend[Any]):
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
