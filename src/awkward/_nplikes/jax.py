# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpy_like import UfuncLike
from awkward._typing import Final, cast


@register_nplike
class Jax(ArrayModuleNumpyLike):  # pylint: disable=too-many-ancestors
    is_eager: Final = True
    supports_structured_dtypes: Final = False

    def __init__(self):
        jax = ak.jax.import_jax()
        self._module = jax.numpy

    def prepare_ufunc(self, ufunc: UfuncLike) -> UfuncLike:
        from awkward._connect.jax import get_jax_ufunc

        return get_jax_ufunc(ufunc)

    @property
    def ma(self):
        raise ValueError(
            "JAX arrays cannot have missing values until JAX implements "
            "numpy.ma.MaskedArray"
        )

    @property
    def char(self):
        raise ValueError(
            "JAX arrays cannot do string manipulations until JAX implements "
            "numpy.char"
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    @classmethod
    def is_own_array_type(cls, type_: type) -> bool:
        """
        Args:
            type_: object to test

        Return `True` if the given object is a jax buffer, otherwise `False`.

        """
        return cls.is_array_type(type_) or cls.is_tracer_type(type_)

    @classmethod
    def is_array_type(cls, type_: type) -> bool:
        """
        Args:
            type_: object to test

        Return `True` if the given object is a jax buffer, otherwise `False`.

        """
        module, _, suffix = type_.__module__.partition(".")
        return module == "jaxlib"

    @classmethod
    def is_tracer_type(cls, type_: type) -> bool:
        """
        Args:
            type_: object to test

        Return `True` if the given object is a jax tracer, otherwise `False`.

        """
        module, _, suffix = type_.__module__.partition(".")
        return module == "jax"

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        return True

    def ascontiguousarray(self, x: ArrayLike) -> ArrayLike:
        return x

    def strides(self, x: ArrayLike) -> tuple[int, ...]:
        out: tuple[int, ...] = (x.dtype.itemsize,)
        for item in cast(tuple[int, ...], x.shape[-1:0:-1]):
            out = (item * out[0], *out)
        return out
