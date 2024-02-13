# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

import jax

import awkward as ak
from awkward import _reducers
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._reducers import Reducer
from awkward._typing import Final, Self, TypeVar

np = NumpyMetadata.instance()


_overloads: dict[type[Reducer], type[JAXReducer]] = {}


R = TypeVar("R", bound=Reducer)


def overloads(cls: type[Reducer]):
    def registrar(new_cls: type[R]) -> type[R]:
        _overloads[cls] = new_cls
        return new_cls

    return registrar


class JAXReducer(Reducer):
    @classmethod
    @abstractmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        raise NotImplementedError


@overloads(_reducers.ArgMin)
class ArgMin(JAXReducer):
    name: Final = "argmin"
    needs_position: Final = True
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        raise NotImplementedError

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        raise RuntimeError("Cannot differentiate through argmin")


@overloads(_reducers.ArgMax)
class ArgMax(JAXReducer):
    name: Final = "argmax"
    needs_position: Final = True
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.ArgMax)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        raise RuntimeError("Cannot differentiate through argmax")


@overloads(_reducers.Count)
class Count(JAXReducer):
    name: Final = "count"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Count)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.numpy.ones_like(array.data, dtype=array.dtype)
        result = jax.ops.segment_sum(result, parents.data)

        if np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.CountNonzero)
class CountNonzero(JAXReducer):
    name: Final = "count_nonzero"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.CountNonzero)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        raise RuntimeError("Cannot differentiate through count_nonzero")


@overloads(_reducers.Sum)
class Sum(JAXReducer):
    name: Final = "sum"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Sum)
        return cls()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise TypeError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")

        result = jax.ops.segment_sum(array.data, parents.data)

        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(result, dtype=array.dtype)
            )
        elif np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.Prod)
class Prod(JAXReducer):
    name: Final = "prod"
    preferred_dtype: Final = np.int64
    needs_position: Final = False

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Prod)
        return cls()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # See issue https://github.com/google/jax/issues/9296
        result = jax.numpy.exp(
            jax.ops.segment_sum(jax.numpy.log(array.data), parents.data)
        )

        if np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.Any)
class Any(JAXReducer):
    name: Final = "any"
    preferred_dtype: Final = np.bool_
    needs_position: Final = False

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Any)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.bool_

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_max(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.All)
class All(JAXReducer):
    name: Final = "all"
    preferred_dtype: Final = np.bool_
    needs_position: Final = False

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.All)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.bool_

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.Min)
class Min(JAXReducer):
    name: Final = "min"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Min)
        return cls(reducer.initial)

    @staticmethod
    def _min_initial(initial, type):
        if initial is None:
            if type in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(type).max
            else:
                return np.inf

        return initial

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.minimum(result, self._min_initial(self.initial, array.dtype))

        if np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(
                    result.view(array.dtype), dtype=array.dtype
                ),
                backend=array.backend,
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.Max)
class Max(JAXReducer):
    name: Final = "max"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Max)
        return cls(reducer.initial)

    @staticmethod
    def _max_initial(initial, type):
        if initial is None:
            if type in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(type).min
            else:
                return -np.inf

        return initial

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_max(array.data, parents.data)

        result = jax.numpy.maximum(result, self._max_initial(self.initial, array.dtype))
        if np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(
                    result.view(array.dtype), dtype=array.dtype
                ),
                backend=array.backend,
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


def get_jax_reducer(reducer: Reducer) -> Reducer:
    return _overloads[type(reducer)].from_kernel_reducer(reducer)
