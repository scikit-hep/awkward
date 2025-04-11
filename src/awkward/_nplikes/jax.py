# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpy_like import UfuncLike
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual
from awkward._typing import Final, cast


@register_nplike
class Jax(ArrayModuleNumpyLike):
    is_eager: Final = True
    supports_structured_dtypes: Final = False
    supports_virtual_arrays: Final = True

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
            "JAX arrays cannot do string manipulations until JAX implements numpy.char"
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

    def is_currently_tracing(self) -> bool:
        jax = ak.jax.import_jax()
        return isinstance(self._module.array(1) + 1, jax.core.Tracer)

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
        if isinstance(x, VirtualArray) and x.is_materialized:
            return x.materialize()
        else:
            return x

    def strides(self, x: ArrayLike) -> tuple[int, ...]:
        out: tuple[int, ...] = (x.dtype.itemsize,)
        for item in cast(tuple[int, ...], x.shape[-1:0:-1]):
            out = (item * out[0], *out)
        return out

    ############################ ufuncs, need to overwrite those because JAX doesn't support `out=` for ufuncs

    def add(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        del maybe_out
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.add(x1, x2)

    def logical_or(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        del maybe_out
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.logical_or(x1, x2)

    def logical_and(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        del maybe_out
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.logical_and(x1, x2)

    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        del maybe_out
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.logical_not(x)

    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        del maybe_out
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.sqrt(x)

    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        del maybe_out
        assert not isinstance(x, PlaceholderArray)
        (x,) = materialize_if_virtual(x)
        return self._module.exp(x)

    def divide(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        del maybe_out
        assert not isinstance(x1, PlaceholderArray)
        assert not isinstance(x2, PlaceholderArray)
        x1, x2 = materialize_if_virtual(x1, x2)
        return self._module.divide(x1, x2)
