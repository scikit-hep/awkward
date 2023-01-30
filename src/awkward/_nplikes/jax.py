# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import numpy

import awkward as ak
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.numpylike import ArrayLike
from awkward.typing import Final


class Jax(ArrayModuleNumpyLike):
    is_eager: Final = True

    def to_rectilinear(self, array, *args, **kwargs):
        return ak.operations.ak_to_jax.to_jax(array, *args, **kwargs)

    def __init__(self):
        jax = ak.jax.import_jax()
        self._module = jax.numpy

    @property
    def ma(self):
        raise ak._errors.wrap_error(
            ValueError(
                "JAX arrays cannot have missing values until JAX implements "
                "numpy.ma.MaskedArray"
            )
        )

    @property
    def char(self):
        raise ak._errors.wrap_error(
            ValueError(
                "JAX arrays cannot do string manipulations until JAX implements "
                "numpy.char"
            )
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    @classmethod
    def is_own_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a jax buffer, otherwise `False`.

        """
        return cls.is_array(obj) or cls.is_tracer(obj)

    @classmethod
    def is_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a jax buffer, otherwise `False`.

        """
        module, _, suffix = type(obj).__module__.partition(".")
        return module == "jaxlib"

    @classmethod
    def is_tracer(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a jax tracer, otherwise `False`.

        """
        module, _, suffix = type(obj).__module__.partition(".")
        return module == "jax"

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        return True

    def ascontiguousarray(
        self, x: ArrayLike, *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        if dtype is not None:
            return x.astype(dtype)
        else:
            return x
