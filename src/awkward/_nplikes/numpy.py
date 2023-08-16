# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import numpy

from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpylike import ArrayLike, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._typing import Final, Literal

np = NumpyMetadata.instance()


@register_nplike
class Numpy(ArrayModuleNumpyLike):
    is_eager: Final = True
    supports_structured_dtypes: Final = True

    def __init__(self):
        self._module = numpy

    @property
    def ma(self):
        return self._module.ma

    @property
    def char(self):
        return self._module.char

    @property
    def ndarray(self):
        return self._module.ndarray

    @classmethod
    def is_own_array_type(cls, type_) -> bool:
        """
        Args:
            type_: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        return issubclass(type_, numpy.ndarray)

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        if isinstance(x, PlaceholderArray):
            return True
        else:
            return x.flags["C_CONTIGUOUS"]

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ):
        assert not isinstance(x, PlaceholderArray)
        return numpy.packbits(x, axis=axis, bitorder=bitorder)

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ):
        assert not isinstance(x, PlaceholderArray)
        return numpy.unpackbits(x, axis=axis, count=count, bitorder=bitorder)
