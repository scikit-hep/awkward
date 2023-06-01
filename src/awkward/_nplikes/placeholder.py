# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from math import prod

from awkward._nplikes.numpylike import ArrayLike, NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._typing import Self

np = NumpyMetadata.instance()


class PlaceholderArray(ArrayLike):
    def __init__(
        self, nplike: NumpyLike, shape: tuple[ShapeItem, ...], dtype: np.dtype
    ):
        self._nplike = nplike
        self._shape = shape
        self._dtype = dtype

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> ShapeItem:
        return prod(self._shape)

    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        out = (self._dtype.itemsize,)
        for item in reversed(self._shape):
            out = (item * out[0], *out)
        return out

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize

    @property
    def T(self):
        return type(self)(self._nplike, self._dtype, self._shape[::-1])

    def view(self, dtype: dtype) -> Self:
        dtype = np.dtype(dtype)
        if (
            self.itemsize != dtype.itemsize
            and len(self._shape) >= 1
            and self._shape[-1] is not None
        ):
            last = int(
                round(self._shape[-1] * self.itemsize / np.dtype(dtype).itemsize)
            )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        return self._new(
            dtype, shape=shape, form_key=self._form_key, report=self._report
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            if self._shape[0] is unknown_length:
                return type(self)(self._nplike, self._shape, self._dtype)
            else:
                start, stop, step = index.indices(self._shape[0])
                new_shape = ((stop - start) // step,)
                return type(self)(self._nplike, new_shape, self._dtype)
        else:
            raise TypeError(
                f"{type(self).__name__} supports only trivial slices, not {type(index).__name__}"
            )

    def __setitem__(self, key, value):
        raise RuntimeError

    def __bool__(self) -> bool:
        raise RuntimeError

    def __int__(self) -> int:
        raise RuntimeError

    def __index__(self) -> int:
        raise RuntimeError

    def __len__(self) -> int:
        return self._shape[0]

    def __add__(self, other):
        raise RuntimeError

    def __and__(self, other):
        raise RuntimeError

    def __eq__(self, other):
        raise RuntimeError

    def __floordiv__(self, other):
        raise RuntimeError

    def __ge__(self, other):
        raise RuntimeError

    def __gt__(self, other):
        raise RuntimeError

    def __invert__(self):
        raise RuntimeError

    def __le__(self, other):
        raise RuntimeError

    def __lt__(self, other):
        raise RuntimeError

    def __mul__(self, other):
        raise RuntimeError

    def __or__(self, other):
        raise RuntimeError

    def __sub__(self, other):
        raise RuntimeError

    def __truediv__(self, other):
        raise RuntimeError

    __iter__ = None
