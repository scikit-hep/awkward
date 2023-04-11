# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import awkward_cpp

import awkward as ak
from awkward._backends.backend import Backend, KernelKeyType
from awkward._kernels import CupyKernel, JaxKernel, NumpyKernel, TypeTracerKernel
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.typetracer import MaybeNone, TypeTracer, TypeTracerArray
from awkward._typing import Final, TypeVar

np = NumpyMetadata.instance()
numpy = Numpy.instance()


T = TypeVar("T", covariant=True)


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
            raise AssertionError(f"CuPyKernel not found: {index!r}")


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
            raise ValueError(f"unsupported ufunc method {method} called")

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

    def __getitem__(self, index: KernelKeyType) -> TypeTracerKernel:
        return TypeTracerKernel(index)

    def _coerce_ufunc_argument(self, x):
        if isinstance(x, TypeTracerArray):
            if x.ndim == 0:
                return numpy.empty((0,), dtype=x.dtype)
            else:
                return numpy.empty((0,) + x.shape[1:], dtype=x.dtype)
        elif isinstance(x, MaybeNone):
            return self._coerce_ufunc_argument(x.content)
        else:
            return x

    def apply_ufunc(self, ufunc, method, args, kwargs):
        shape = None
        numpy_args = []

        for x in args:
            if isinstance(x, TypeTracerArray):
                x.touch_data()
                shape = x.shape

            numpy_args.append(self._coerce_ufunc_argument(x))

        assert shape is not None
        tmp = getattr(ufunc, method)(*numpy_args, **kwargs)
        return self._typetracer.empty((shape[0],) + tmp.shape[1:], dtype=tmp.dtype)
