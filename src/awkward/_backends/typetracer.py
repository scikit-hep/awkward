# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from awkward._backends.backend import Backend, KernelKeyType
from awkward._backends.dispatch import register_backend
from awkward._kernels import TypeTracerKernel
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.typetracer import MaybeNone, TypeTracer, TypeTracerArray
from awkward._typing import Final

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@register_backend(TypeTracer)
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
