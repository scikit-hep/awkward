# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from abc import abstractmethod
from typing import overload

import numpy

import awkward as ak
from awkward._singleton import Singleton
from awkward.typing import Literal, Protocol, Self, SupportsIndex, SupportsInt, TypeVar


class ArrayLike(Protocol):
    @property
    @abstractmethod
    def dtype(self) -> dtype:
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[SupportsInt, ...]:
        ...

    @property
    @abstractmethod
    def size(self) -> SupportsInt:
        ...

    @property
    @abstractmethod
    def T(self) -> Self:
        ...

    @overload
    def __getitem__(
        self, index: SupportsIndex
    ) -> int | float | complex | str | bytes | bytes:
        ...

    @overload
    def __getitem__(
        self, index: slice | Ellipsis | tuple[SupportsIndex | slice | Ellipsis, ...]
    ) -> Self:
        ...

    @abstractmethod
    def __getitem__(self, index) -> Self:
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        ...

    @abstractmethod
    def __int__(self) -> int:
        ...

    @abstractmethod
    def __index__(self) -> int:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def view(self, dtype: dtype) -> Self:
        ...


class NumpyMetadata(Singleton):
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    longlong = numpy.longlong
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    str_ = numpy.str_
    bytes_ = numpy.bytes_

    intp = numpy.intp
    integer = numpy.integer
    signedinteger = numpy.signedinteger
    unsignedinteger = numpy.unsignedinteger
    floating = numpy.floating
    complexfloating = numpy.complexfloating
    number = numpy.number
    object_ = numpy.object_
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    errstate = numpy.errstate
    newaxis = numpy.newaxis

    ndarray = numpy.ndarray

    nan = numpy.nan
    inf = numpy.inf

    nat = numpy.datetime64("NaT")
    datetime_data = numpy.datetime_data

    @property
    def issubdtype(self):
        return numpy.issubdtype

    AxisError = numpy.AxisError


if hasattr(numpy, "float16"):
    NumpyMetadata.float16 = numpy.float16

if hasattr(numpy, "float128"):
    NumpyMetadata.float128 = numpy.float128

if hasattr(numpy, "complex256"):
    NumpyMetadata.complex256 = numpy.complex256

if hasattr(numpy, "datetime64"):
    NumpyMetadata.datetime64 = numpy.datetime64

if hasattr(numpy, "timedelta64"):
    NumpyMetadata.timedelta64 = numpy.timedelta64


np = NumpyMetadata.instance()


class NumpyLike(Singleton):
    known_data = True
    known_shape = True
    known_dtype = True
    is_eager = True

    ############################ array creation

    def asarray(
        self,
        obj,
        *,
        dtype: numpy.dtype | None = None,
        copy: bool | None = None,
    ) -> ArrayLike:
        if copy:
            return self._module.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            return self._module.asarray(obj, dtype=dtype)
        else:
            if getattr(obj, "dtype", dtype) != dtype:
                raise ak._errors.wrap_error(
                    ValueError(
                        "asarray was called with copy=False for an array of a different dtype"
                    )
                )
            else:
                return self._module.asarray(obj, dtype=dtype)

    def ascontiguousarray(
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.ascontiguousarray(x, dtype=dtype)

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> ArrayLike:
        return self._module.frombuffer(buffer, dtype=dtype, count=count)

    def zeros(self, shape: int | tuple[int, ...], *, dtype: np.dtype) -> ArrayLike:
        return self._module.zeros(shape, dtype=dtype)

    def ones(self, shape: int | tuple[int, ...], *, dtype: np.dtype) -> ArrayLike:
        return self._module.ones(shape, dtype=dtype)

    def empty(self, shape: int | tuple[int, ...], *, dtype: np.dtype) -> ArrayLike:
        return self._module.empty(shape, dtype=dtype)

    def full(
        self, shape: int | tuple[int, ...], fill_value, *, dtype: np.dtype
    ) -> ArrayLike:
        return self._module.full(shape, fill_value, dtype=dtype)

    def zeros_like(self, x: ArrayLike, *, dtype: np.dtype | None = None) -> ArrayLike:
        return self._module.zeros_like(x, dtype=dtype)

    def ones_like(self, x: ArrayLike, *, dtype: np.dtype | None = None) -> ArrayLike:
        return self._module.ones_like(x, dtype=dtype)

    def full_like(
        self, x: ArrayLike, fill_value, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        return self._module.full_like(x, fill_value, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: np.dtype | None = None,
    ) -> ArrayLike:
        return self._module.arange(start, stop, step, dtype=dtype)

    def meshgrid(
        self, *arrays: ArrayLike, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[ArrayLike]:
        return self._module.meshgrid(*arrays, indexing=indexing)

    ############################ testing

    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> bool:
        return self._module.array_equal(x1, x2, equal_nan=equal_nan)

    def searchsorted(
        self,
        x: ArrayLike,
        values: ArrayLike,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.searchsorted(x, values, side=side, sorter=sorter)

    ############################ manipulation

    def broadcast_arrays(self, *arrays: ArrayLike) -> list[ArrayLike]:
        return self._module.broadcast_arrays(*arrays)

    def nonzero(self, x: ArrayLike) -> tuple[ArrayLike, ...]:
        return self._module.nonzero(x)

    def unique_values(self, x: ArrayLike) -> ArrayLike:
        return self._module.unique(
            x,
            return_counts=False,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    def concat(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayLike:
        return self._module.concatenate(arrays, axis=axis, casting="same_kind")

    def repeat(
        self,
        x: ArrayLike,
        repeats: ArrayLike | int,
        *,
        axis: int | None = None,
    ) -> ArrayLike:
        return self._module.repeat(x, repeats=repeats, axis=axis)

    def tile(self, x: ArrayLike, reps: int) -> ArrayLike:
        return self._module.tile(x, reps)

    def stack(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int = 0,
    ) -> ArrayLike:
        arrays = [x for x in arrays]
        return self._module.stack(arrays, axis=axis)

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        return self._module.packbits(x, axis=axis, bitorder=bitorder)

    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        return self._module.unpackbits(x, axis=axis, count=count, bitorder=bitorder)

    def broadcast_to(self, x: ArrayLike, shape: tuple[SupportsInt, ...]) -> ArrayLike:
        return self._module.broadcast_to(x, shape)

    ############################ ufuncs

    def add(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.add(x1, x2, out=maybe_out)

    def logical_or(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.logical_or(x1, x2, out=maybe_out)

    def logical_and(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.logical_and(x1, x2, out=maybe_out)

    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.logical_not(x, out=maybe_out)

    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        return self._module.sqrt(x, out=maybe_out)

    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        return self._module.exp(x, out=maybe_out)

    def divide(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        return self._module.divide(x1, x2, out=maybe_out)

    ############################ almost-ufuncs

    def nan_to_num(
        self,
        x: ArrayLike,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayLike:
        return self._module.nan_to_num(
            x, copy=copy, nan=nan, posinf=posinf, neginf=neginf
        )

    def isclose(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayLike:
        return self._module.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isnan(self, x: ArrayLike) -> ArrayLike:
        return self._module.isnan(x)

    def all(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.all(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.any(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.min(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.max(x, axis=axis, keepdims=keepdims, out=maybe_out)

    def count_nonzero(
        self, x: ArrayLike, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayLike:
        return self._module.count_nonzero(x, axis=axis, keepdims=keepdims)

    def cumsum(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        return self._module.cumsum(x, axis=axis, out=maybe_out)

    def array_str(
        self,
        x: ArrayLike,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        return self._module.array_str(
            x,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
        )

    def can_cast(self, from_: np.dtype | ArrayLike, to: np.dtype | ArrayLike) -> bool:
        return self._module.can_cast(from_, to, casting="same_kind")

    def raw(self, array: ArrayLike, nplike: NumpyLike) -> ArrayLike:
        raise ak._errors.wrap_error(NotImplementedError)

    @classmethod
    def is_own_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        raise ak._errors.wrap_error(NotImplementedError)

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        raise ak._errors.wrap_error(NotImplementedError)

    def to_rectilinear(self, array: ArrayLike) -> ArrayLike:
        raise ak._errors.wrap_error(NotImplementedError)


class Numpy(NumpyLike):
    def to_rectilinear(self, array, *args, **kwargs):
        return ak.operations.ak_to_numpy.to_numpy(array, *args, **kwargs)

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

    def raw(self, array: ArrayLike, nplike: NumpyLike) -> ArrayLike:
        if isinstance(nplike, Numpy):
            return array
        elif isinstance(nplike, Cupy):
            return nplike.asarray(array, order="C")
        elif isinstance(nplike, ak._typetracer.TypeTracer):
            return nplike.asarray(array)
        elif isinstance(nplike, Jax):
            jax = Jax.instance()
            return jax.asarray(array)
        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax"
                )
            )

    @classmethod
    def is_own_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        return isinstance(obj, numpy.ndarray)

    def is_c_contiguous(self, x: ArrayLike) -> bool:
        return x.flags["C_CONTIGUOUS"]

    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ):
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


class Cupy(NumpyLike):
    is_eager = False

    def to_rectilinear(self, array, *args, **kwargs):
        return ak.operations.ak_to_cupy.to_cupy(array, *args, **kwargs)

    def __init__(self):
        import awkward._connect.cuda  # noqa: F401

        self._module = ak._connect.cuda.import_cupy("Awkward Arrays with CUDA")

    @property
    def ma(self):
        raise ak._errors.wrap_error(
            ValueError(
                "CUDA arrays cannot have missing values until CuPy implements "
                "numpy.ma.MaskedArray"
            )
        )

    @property
    def char(self):
        raise ak._errors.wrap_error(
            ValueError(
                "CUDA arrays cannot do string manipulations until CuPy implements "
                "numpy.char"
            )
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def raw(self, array: ArrayLike, nplike: NumpyLike) -> ArrayLike:
        if isinstance(nplike, Cupy):
            return array
        elif isinstance(nplike, Numpy):
            return nplike.asarray(array.get(), dtype=array.dtype, order="C")
        elif isinstance(nplike, ak._typetracer.TypeTracer):
            return nplike.asarray(array, dtype=array.dtype)
        elif isinstance(nplike, Jax):
            return nplike.asarray(array.get(), dtype=array.dtype)
        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax"
                )
            )

    def frombuffer(
        self, buffer, *, dtype: np.dtype | None = None, count: int = -1
    ) -> ArrayLike:
        np_array = numpy.frombuffer(buffer, dtype=dtype, count=count)
        return self._module.asarray(np_array)

    def array_equal(self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False):
        if x1.shape != x2.shape:
            return False
        else:
            return self._module.all(x1 - x2 == 0)

    def repeat(
        self, x: ArrayLike, repeats: ArrayLike | int, *, axis: int | None = None
    ):
        if axis is not None:
            raise ak._errors.wrap_error(
                NotImplementedError(f"repeat for CuPy with axis={axis!r}")
            )
        # https://github.com/cupy/cupy/issues/3849
        if isinstance(repeats, self._module.ndarray):
            all_stops = self._module.cumsum(repeats)
            parents = self._module.zeros(all_stops[-1].item(), dtype=int)
            stops, stop_counts = self._module.unique(all_stops[:-1], return_counts=True)
            parents[stops] = stop_counts
            self._module.cumsum(parents, out=parents)
            return x[parents]
        else:
            return self._module.repeat(x, repeats=repeats)

    # For all reducers: https://github.com/cupy/cupy/issues/3819

    def all(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        out = self._module.all(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        out = self._module.any(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def count_nonzero(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        out = self._module.count_nonzero(x, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        out = self._module.min(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        out = self._module.max(x, axis=axis, out=maybe_out)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def array_str(
        self, array, max_line_width=None, precision=None, suppress_small=None
    ):
        return self._module.array_str(array, max_line_width, precision, suppress_small)

    @classmethod
    def is_own_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a cupy buffer, otherwise `False`.

        """
        module, _, suffix = type(obj).__module__.partition(".")
        return module == "cupy"


class Jax(NumpyLike):
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

    def raw(self, array: ArrayLike, nplike: NumpyLike) -> ArrayLike:
        if isinstance(nplike, Jax):
            return array
        elif isinstance(nplike, ak._nplikes.Cupy):
            return nplike.asarray(array)
        elif isinstance(nplike, ak._nplikes.Numpy):
            return nplike.asarray(array)
        elif isinstance(nplike, ak._typetracer.TypeTracer):
            return nplike.asarray(array, dtype=array.dtype)
        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax",
                )
            )

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
        self, x: ArrayLike, *, dtype: np.dtype | None = None
    ) -> ArrayLike:
        if dtype is not None:
            return x.astype(dtype)
        else:
            return x


# Temporary sentinel marking "argument not given"
_UNSET = object()

D = TypeVar("D")


def nplike_of(*arrays, default: D = _UNSET) -> NumpyLike | D:
    """
    Args:
        *arrays: iterable of possible array objects
        default: default NumpyLike instance if no array objects found

    Return the #ak._nplikes.NumpyLike that is best-suited to operating upon the given
    iterable of arrays. Return an instance of the `default_cls` if no known array types
    are found.
    """
    nplikes: set[NumpyLike] = set()
    nplike_classes = (Numpy, Cupy, Jax, ak._typetracer.TypeTracer)
    for array in arrays:
        if hasattr(array, "layout"):
            array = array.layout

        # Layout objects
        if hasattr(array, "backend"):
            nplikes.add(array.backend.nplike)

        # Index objects
        elif hasattr(array, "nplike"):
            nplikes.add(array.nplike)

        # Other e.g. nplike arrays
        else:
            for cls in nplike_classes:
                if cls.is_own_array(array):
                    nplikes.add(cls.instance())
                    break

    if nplikes == set():
        if default is _UNSET:
            return Numpy.instance()
        else:
            return default
    elif len(nplikes) == 1:
        return next(iter(nplikes))
    else:
        # We allow typetracers to mix with other nplikes, and take precedence
        for nplike in nplikes:
            if not (nplike.known_data and nplike.known_shape):
                return nplike

        raise ak._errors.wrap_error(
            ValueError(
                """attempting to use arrays with more than one backend in the same operation; use
#ak.to_backend to coerce the arrays to the same backend."""
            )
        )
