# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward.typing import Final


class JAX(ArrayModuleNumpyLike):
    """
    A concrete class importing `NumpyModuleLike` for `JAX`
    """

    is_eager: Final[bool] = True

    @property
    def array_module(self):
        import jax.numpy

        return jax.numpy

    @classmethod
    def is_tracer(cls, obj: object) -> bool:
        raise ak._errors.wrap_error(NotImplementedError)
