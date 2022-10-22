# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from numbers import Real

import jax

import awkward as ak
from awkward import reducers
from awkward.reducers import DTypeLike, Reducer

np = ak.nplikes.NumpyMetadata.instance()


_overloads = {}


def overloads(reducer_cls: type[Reducer]):
    def wrapper(overload_cls):
        _overloads[reducer_cls] = overload_cls
        return overload_cls

    return wrapper


def get_jax_reducer(reducer: Reducer) -> Reducer:
    if isinstance(reducer, type):
        raise ak._errors.wrap_error(TypeError("expected reducer instance"))

    reducer_cls = type(reducer)
    jax_reducer_cls = _overloads[reducer_cls]
    return jax_reducer_cls.from_overloaded(reducer)


class JaxReducer(Reducer):
    @classmethod
    def from_overloaded(cls, overloaded: Reducer):
        return cls()


@overloads(reducers.ArgMin)
class ArgMin(JaxReducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        raise ak._errors.wrap_error(RuntimeError("Cannot differentiate through argmin"))


@overloads(reducers.ArgMax)
class ArgMax(JaxReducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        raise ak._errors.wrap_error(RuntimeError("Cannot differentiate through argmax"))


@overloads(reducers.Count)
class Count(JaxReducer):
    name = "count"
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        raise ak._errors.wrap_error(
            RuntimeError("Cannot differentiate through count_zero")
        )


@overloads(reducers.CountNonzero)
class CountNonzero(JaxReducer):
    name = "count_nonzero"
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        raise ak._errors.wrap_error(
            RuntimeError("Cannot differentiate through count_nonzero")
        )


@overloads(reducers.Sum)
class Sum(JaxReducer):
    name = "sum"
    preferred_dtype = np.float64

    def apply(self, array, parents, outlength):

        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ak._errors.wrap_error(
                TypeError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")
            )

        result = jax.ops.segment_sum(array.data, parents.data)

        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(array.nplike.asarray(result, array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result, nplike=array.nplike)


@overloads(reducers.Prod)
class Prod(JaxReducer):
    name = "prod"
    preferred_dtype = np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        # See issue https://github.com/google/jax/issues/9296
        result = jax.numpy.exp(
            jax.ops.segment_sum(jax.numpy.log(array.data), parents.data)
        )

        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype), nplike=array.nplike)
        else:
            return ak.contents.NumpyArray(result, nplike=array.nplike)


@overloads(reducers.Any)
class Any(JaxReducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_max(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, nplike=array.nplike)


@overloads(reducers.All)
class All(JaxReducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, nplike=array.nplike)


@overloads(reducers.Min)
class Min(JaxReducer):
    name = "min"
    preferred_dtype = np.float64

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    @classmethod
    def from_overloaded(cls, overloaded: reducers.Min):
        return cls(overloaded.initial)

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        dtype = np.dtype(dtype)
        if self._initial is None:
            if dtype in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(dtype).max
            elif dtype.kind == "M":
                unit, _ = np.datetime_data(dtype)
                return np.datetime64(np.iinfo(np.int64).max, unit)
            elif dtype.kind == "m":
                unit, _ = np.datetime_data(dtype)
                return np.timedelta64(np.iinfo(np.int64).max, unit)
            else:
                return np.inf

        return self._initial

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.minimum(result, self.identity_for(array.dtype))

        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype),
                nplike=array.nplike,
            )
        else:
            return ak.contents.NumpyArray(result, nplike=array.nplike)


@overloads(reducers.Max)
class Max(JaxReducer):
    name = "max"
    preferred_dtype = np.float64

    def __init__(self, initial):
        self._initial = initial

    @classmethod
    def from_overloaded(cls, overloaded: reducers.Max):
        return cls(overloaded.initial)

    def identity_for(self, dtype: DTypeLike | None):
        dtype = np.dtype(dtype)

        if self._initial is None:
            if dtype in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(dtype).min
            elif dtype.kind == "M":
                unit, _ = np.datetime_data(dtype)
                return np.datetime64(np.iinfo(np.int64).min + 1, unit)
            elif dtype.kind == "m":
                unit, _ = np.datetime_data(dtype)
                return np.timedelta64(np.iinfo(np.int64).min + 1, unit)
            else:
                return -np.inf

        return self._initial

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_max(array.data, parents.data)

        result = jax.numpy.maximum(result, self.identity_for(array.dtype))
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype),
                nplike=array.nplike,
            )
        else:
            return ak.contents.NumpyArray(result, nplike=array.nplike)
