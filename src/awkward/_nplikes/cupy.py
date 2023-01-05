# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.numpy import ArrayModuleNumpyLike
from awkward.typing import Final


class Cupy(ArrayModuleNumpyLike):

    is_eager: Final[bool] = False

    @property
    def array_module(self):
        import cupy

        return cupy

    @classmethod
    def is_own_array(cls, x) -> bool:
        module, _, suffix = type(x).__module__.partition(".")
        return module == "cupy"
