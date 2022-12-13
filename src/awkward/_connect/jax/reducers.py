# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import jax

import awkward as ak
from awkward._reducers import Reducer

np = ak._nplikes.NumpyMetadata.instance()


class ArgMin(Reducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(RuntimeError("Cannot differentiate through argmin"))


class ArgMax(Reducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(RuntimeError("Cannot differentiate through argmax"))


class Count(Reducer):
    name = "count"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(
            RuntimeError("Cannot differentiate through count_zero")
        )


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(
            RuntimeError("Cannot differentiate through count_nonzero")
        )


class Sum(Reducer):
    name = "sum"
    preferred_dtype = np.float64

    @classmethod
    def apply(cls, array, parents, outlength):

        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ak._errors.wrap_error(
                TypeError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")
            )

        result = jax.ops.segment_sum(array.data, parents.data)

        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(result, array.dtype)
            )
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


class Prod(Reducer):
    name = "prod"
    preferred_dtype = np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        # See issue https://github.com/google/jax/issues/9296
        result = jax.numpy.exp(
            jax.ops.segment_sum(jax.numpy.log(array.data), parents.data)
        )

        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


class Any(Reducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_max(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, backend=array.backend)


class All(Reducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, backend=array.backend)


class Min(Reducer):
    name = "min"
    preferred_dtype = np.float64
    initial = None

    def __init__(self, initial):
        type(self).initial = initial

    def __del__(self):
        type(self).initial = None

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

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.minimum(result, cls._min_initial(cls.initial, array.dtype))

        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.backend.nplike.array(result.view(array.dtype), array.dtype),
                backend=array.backend,
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


class Max(Reducer):
    name = "max"
    preferred_dtype = np.float64
    initial = None

    def __init__(self, initial):
        type(self).initial = initial

    def __del__(self):
        type(self).initial = None

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

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_max(array.data, parents.data)

        result = jax.numpy.maximum(result, cls._max_initial(cls.initial, array.dtype))
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.backend.nplike.array(result.view(array.dtype), array.dtype),
                backend=array.backend,
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


def get_jax_reducer(reducer: Reducer) -> Reducer:
    if isinstance(reducer, type):
        return globals()[reducer.__name__]
    else:
        return globals()[type(reducer).__name__]
