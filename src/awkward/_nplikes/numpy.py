# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from typing import Any

import numpy

from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.numpylike import ErrorStateLiteral
from awkward.typing import ContextManager, Final


class Numpy(ArrayModuleNumpyLike):
    """
    A concrete class importing `NumpyModuleLike` for `numpy`
    """

    is_eager: Final[bool] = True

    array_module: Final[Any] = numpy

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return isinstance(obj, numpy.ndarray)

    def error_state(
        self,
        **kwargs: ErrorStateLiteral,
    ) -> ContextManager:
        return numpy.errstate(**kwargs)
