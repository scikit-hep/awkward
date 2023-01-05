# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import jax

import awkward as ak
from awkward._nplikes import metadata
from awkward._reducers import Reducer


class ArgMin(Reducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = metadata.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(RuntimeError("Cannot differentiate through argmin"))


class ArgMax(Reducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = metadata.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(RuntimeError("Cannot differentiate through argmax"))


class Count(Reducer):
    name = "count"
    preferred_dtype = metadata.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(
            RuntimeError("Cannot differentiate through count_zero")
        )


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = metadata.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._errors.wrap_error(
            RuntimeError("Cannot differentiate through count_nonzero")
        )


class Sum(Reducer):
    name = "sum"
    preferred_dtype = metadata.float64

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
                array.backend.nplike.asarray(result, dtype=array.dtype)
            )
        elif metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


class Prod(Reducer):
    name = "prod"
    preferred_dtype = metadata.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        # See issue https://github.com/google/jax/issues/9296
        result = jax.numpy.exp(
            jax.ops.segment_sum(jax.numpy.log(array.data), parents.data)
        )

        if metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


class Any(Reducer):
    name = "any"
    preferred_dtype = metadata.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_max(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, backend=array.backend)


class All(Reducer):
    name = "all"
    preferred_dtype = metadata.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.asarray(result, dtype=bool)

        return ak.contents.NumpyArray(result, backend=array.backend)


class Min(Reducer):
    name = "min"
    preferred_dtype = metadata.float64
    initial = None

    def __init__(self, initial):
        type(self).initial = initial

    def __del__(self):
        type(self).initial = None

    @staticmethod
    def _min_initial(initial, type):
        if initial is None:
            if metadata.isdtype(type, "integral"):
                return metadata.iinfo(type).max
            else:
                return metadata.inf

        return initial

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_min(array.data, parents.data)
        result = jax.numpy.minimum(result, cls._min_initial(cls.initial, array.dtype))

        if metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(
                    result.view(array.dtype), dtype=array.dtype
                ),
                backend=array.backend,
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


class Max(Reducer):
    name = "max"
    preferred_dtype = metadata.float64
    initial = None

    def __init__(self, initial):
        type(self).initial = initial

    def __del__(self):
        type(self).initial = None

    @staticmethod
    def _max_initial(initial, type):
        if initial is None:
            if metadata.isdtype(type, "integral"):
                return metadata.iinfo(type).min
            else:
                return -metadata.inf

        return initial

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)

        result = jax.ops.segment_max(array.data, parents.data)

        result = jax.numpy.maximum(result, cls._max_initial(cls.initial, array.dtype))
        if metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(
                    result.view(array.dtype), dtype=array.dtype
                ),
                backend=array.backend,
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


def get_jax_reducer(reducer: Reducer) -> Reducer:
    if isinstance(reducer, type):
        return globals()[reducer.__name__]
    else:
        return globals()[type(reducer).__name__]
