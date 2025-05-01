# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy

import awkward as ak
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._operators import NDArrayOperatorsMixin
from awkward._regularize import is_integer
from awkward._typing import TYPE_CHECKING, Any, Callable, DType, Self
from awkward._util import Sentinel

np = NumpyMetadata.instance()

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


UNMATERIALIZED = Sentinel("UNMATERIALIZED", None)


def assert_never():
    raise AssertionError("this shape_generator should never be run!")


def materialize_if_virtual(*args: Any) -> tuple[Any, ...]:
    """
    A little helper function to materialize all virtual arrays in a list of arrays.
    """
    return tuple(
        arg.materialize() if isinstance(arg, VirtualArray) else arg for arg in args
    )


class VirtualArray(NDArrayOperatorsMixin, ArrayLike):
    """
    Implements a virtual array to be used as a buffer inside layouts.
    Virtual arrays are tied to specific nplikes.
    The arrays are generated via a generator function that is passed to the constructor.
    They optionally accept a shape generator function that is called when the shape of the array is unknown.
    If it doesn't exist, the shape is generated from the materialized array.
    All virtual arrays also required to have a known dtype and shape but can contain `unknown_length` dimensions.
    Some operations (such as trivial slicing) maintain virtualness and return a new virtual array.
    Others are required to access the underlying data of the array and therefore materialize it.
    The materialized arrays are cached on themselves in the `_array` property.
    Subsequent virtual arrays that originate from some virtual array will hit the cache of their parents if there is any.
    """

    def __init__(
        self,
        nplike: NumpyLike,
        shape: tuple[ShapeItem, ...],
        dtype: DTypeLike,
        generator: Callable[[], ArrayLike],
        shape_generator: Callable[[], tuple[ShapeItem, ...]] | None = None,
    ) -> None:
        if not nplike.supports_virtual_arrays:
            raise TypeError(
                f"The nplike {type(nplike)} does not support virtual arrays"
            )
        if not all(is_integer(dim) or dim is unknown_length for dim in shape):
            raise TypeError(
                f"Only shapes of integer dimensions or unknown_length are supported for {type(self).__name__}. Received shape {shape}"
            )

        # array metadata
        self._nplike = nplike
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._array: Sentinel | ArrayLike = UNMATERIALIZED
        self._generator = generator
        self._shape_generator = shape_generator

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> tuple[ShapeItem, ...]:
        self.get_shape()
        return self._shape

    @property
    def inner_shape(self) -> tuple[ShapeItem, ...]:
        if len(self._shape) > 1:
            return self.shape[1:]
        return self._shape[1:]

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> ShapeItem:
        size: ShapeItem = 1
        for item in self.shape:
            size *= item
        return size

    @property
    def nbytes(self) -> ShapeItem:
        size: ShapeItem = 1
        for item in self._shape:
            size *= item
        return size * self._dtype.itemsize

    @property
    def strides(self) -> tuple[ShapeItem, ...]:
        return self.materialize().strides  # type: ignore[attr-defined]

    def get_shape(self) -> None:
        if any(dim is unknown_length for dim in self._shape):
            if self._shape_generator is not None:
                shape = self._shape_generator()
            else:
                shape = self.materialize().shape
            if len(shape) != len(self._shape):
                raise ValueError(
                    f"{type(self).__name__} had shape {self._shape} before materialization while the materialized array has shape {shape}"
                )
            for expected_dim, actual_dim in zip(self._shape, shape):
                if expected_dim is not unknown_length and expected_dim != actual_dim:
                    raise ValueError(
                        f"{type(self).__name__} had shape {self._shape} before materialization while the materialized array has shape {shape}"
                    )
            if not all(is_integer(dim) for dim in shape):
                raise ValueError(
                    f"Only shapes of integer dimensions are supported for materialized shapes. Received shape {shape}"
                )
            self._shape = tuple(map(int, shape))
            self._shape_generator = assert_never

    def materialize(self) -> ArrayLike:
        if self._array is UNMATERIALIZED:
            array = self._nplike.asarray(self._generator())
            if len(self._shape) != len(array.shape):
                raise ValueError(
                    f"{type(self).__name__} had shape {self._shape} before materialization while the materialized array has shape {array.shape}"
                )
            for expected_dim, actual_dim in zip(self._shape, array.shape):
                if expected_dim is not unknown_length and expected_dim != actual_dim:
                    raise ValueError(
                        f"{type(self).__name__} had shape {self._shape} before materialization while the materialized array has shape {array.shape}"
                    )
            if self._dtype != array.dtype:
                raise ValueError(
                    f"{type(self).__name__} had dtype {self._dtype} before materialization while the materialized array has dtype {array.dtype}"
                )
            self._shape = array.shape
            self._array = array
        return self._array  # type: ignore[return-value]

    def dematerialize(self) -> None:
        self._array = UNMATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return self._array is not UNMATERIALIZED

    @property
    def T(self):
        if self._array is not UNMATERIALIZED:
            return self._array.T

        return type(self)(
            self._nplike,
            self._shape[::-1],
            self._dtype,
            lambda: self.materialize().T,
            lambda: self.shape[::-1],
        )

    def view(self, dtype: DTypeLike) -> Self:
        dtype = np.dtype(dtype)

        if self._array is not UNMATERIALIZED:
            return self.materialize().view(dtype)  # type: ignore[return-value]

        if len(self.shape) >= 1:
            last, remainder = divmod(
                self.shape[-1] * self._dtype.itemsize, dtype.itemsize
            )
            if remainder != 0:
                raise ValueError(
                    "new size of array with larger dtype must be a "
                    "divisor of the total size in bytes (of the last axis of the array)"
                )
            shape = self.shape[:-1] + (last,)
        else:
            shape = self.shape

        return type(self)(
            self._nplike,
            shape,
            dtype,
            lambda: self.materialize().view(dtype),
            lambda: shape,
        )

    @property
    def nplike(self) -> NumpyLike:
        if not self._nplike.supports_virtual_arrays:
            raise TypeError(
                f"The nplike {type(self._nplike)} does not support virtual arrays"
            )
        return self._nplike

    def copy(self) -> VirtualArray:
        return copy.copy(self)

    def tolist(self) -> list:
        return self.materialize().tolist()  # type: ignore[attr-defined]

    @property
    def ctypes(self):
        if isinstance((self._nplike), ak._nplikes.cupy.Cupy):
            raise AttributeError("Cupy ndarrays do not have a ctypes attribute")
        return self.materialize().ctypes

    @property
    def data(self):
        return self.materialize().data

    def unsafe_buffer_pointer(self):
        assert isinstance(self._nplike, ak._nplikes.jax.Jax)
        return self.materialize().unsafe_buffer_pointer()

    def byteswap(self, inplace=False):
        if self._array is not UNMATERIALIZED:
            return self._array.byteswap(inplace=inplace)

        return type(self)(
            self._nplike,
            self._shape,
            self._dtype,
            lambda: self.materialize().byteswap(inplace=inplace),
            lambda: self.shape,
        )

    def tobytes(self, order="C") -> bytes:
        return self.materialize().tobytes(order)  # type: ignore[attr-defined]

    def __copy__(self) -> VirtualArray:
        new_virtual = type(self)(
            self._nplike,
            self._shape,
            self._dtype,
            self._generator,
            self._shape_generator,
        )
        new_virtual._array = self._array
        return new_virtual

    def __deepcopy__(self, memo) -> VirtualArray:
        new_virtual = type(self)(
            self._nplike,
            self._shape,
            self._dtype,
            lambda: copy.deepcopy(self._generator(), memo),
            self._shape_generator,
        )
        new_virtual._array = (
            copy.deepcopy(self._array, memo)
            if self._array is not UNMATERIALIZED
            else UNMATERIALIZED
        )
        return new_virtual

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.nplike.apply_ufunc(ufunc, method, inputs, kwargs)

    def __repr__(self):
        dtype = repr(self._dtype)
        if self._shape is None:
            shape = ""
        else:
            shape = f", shape={self._shape!r}"
        return f"VirtualArray(array={self._array}, {dtype}{shape})"

    def __str__(self):
        return repr(self) if self._shape else "??"

    def __getitem__(self, index):
        (index,) = materialize_if_virtual(index)
        if self._array is not UNMATERIALIZED:
            return self._array.__getitem__(index)

        if isinstance(index, slice):
            if (
                index.start is unknown_length
                or index.stop is unknown_length
                or index.step is unknown_length
            ):
                raise TypeError(
                    f"{type(self).__name__} does not support slicing with unknown_length while slice {index} was provided"
                )
            else:
                length = self.shape[0]
                start, stop, step = index.indices(length)
                new_length = max(
                    0, (stop - start + (step - (1 if step > 0 else -1))) // step
                )

            return type(self)(
                self._nplike,
                (new_length,) + self.shape[1:],
                self._dtype,
                lambda: self.materialize()[index],
                lambda: (new_length,) + self.shape[1:],
            )
        else:
            return self.materialize().__getitem__(index)

    def __setitem__(self, key, value):
        array = self.materialize()
        (value,) = materialize_if_virtual(value)
        if isinstance(self._nplike, ak._nplikes.jax.Jax):
            self._array = array.at[key].set(value)
        else:
            array.__setitem__(key, value)

    def __bool__(self) -> bool:
        array = self.materialize()
        return bool(array)

    def __int__(self) -> int:
        array = self.materialize()
        if len(array.shape) == 0:
            return int(array)
        raise TypeError("Only scalar arrays can be converted to an int")

    def __index__(self) -> int:
        array = self.materialize()
        if len(array.shape) == 0:
            return int(array)
        raise TypeError("Only scalar arrays can be used as an index")

    def __len__(self) -> int:
        if len(self._shape) == 0:
            raise TypeError("len() of unsized object")
        return int(self.shape[0])

    def __iter__(self):
        array = self.materialize()
        return iter(array)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.materialize().__dlpack_device__()  # type: ignore[attr-defined]

    def __dlpack__(self, stream: Any = None) -> Any:
        return self.materialize().__dlpack__(stream=stream)  # type: ignore[attr-defined]
