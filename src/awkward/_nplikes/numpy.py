# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import numpy

import awkward as ak
from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.dispatch import register_nplike
from awkward._nplikes.numpylike import ArrayLike, NumpyMetadata
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._typing import Final, Literal

np = NumpyMetadata.instance()


@register_nplike
class Numpy(ArrayModuleNumpyLike):
    is_eager: Final = True

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
        if ak._util.numpy_at_least("1.17.0"):
            return numpy.packbits(x, axis=axis, bitorder=bitorder)
        else:
            assert axis is None, "unsupported argument value for axis given"
            if bitorder == "little":
                if len(x) % 8 == 0:
                    ready_to_pack = x
                else:
                    ready_to_pack = numpy.empty(
                        int(numpy.ceil(len(x) / 8.0)) * 8,
                        dtype=x.dtype,
                    )
                    ready_to_pack[: len(x)] = x
                    ready_to_pack[len(x) :] = 0
                return numpy.packbits(ready_to_pack.reshape(-1, 8)[:, ::-1].reshape(-1))
            else:
                assert bitorder == "bit"
                return numpy.packbits(x)

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ):
        assert not isinstance(x, PlaceholderArray)
        if ak._util.numpy_at_least("1.17.0"):
            return numpy.unpackbits(x, axis=axis, count=count, bitorder=bitorder)
        else:
            assert axis is None, "unsupported argument value for axis given"
            assert count is None, "unsupported argument value for count given"
            ready_to_bitswap = numpy.unpackbits(x)
            if bitorder == "little":
                return ready_to_bitswap.reshape(-1, 8)[:, ::-1].reshape(-1)
            else:
                assert bitorder == "bit"
                return ready_to_bitswap
