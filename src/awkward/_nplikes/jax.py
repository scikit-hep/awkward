# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward.typing import Final


class Jax(ArrayModuleNumpyLike):
    """
    A concrete class importing `NumpyModuleLike` for `JAX`
    """

    is_eager: Final[bool] = True

    @property
    def array_module(self):
        import jax.numpy

        return jax.numpy

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
