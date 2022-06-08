# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._v2._connect.jax import import_jax

np = ak.nplike.NumpyMetadata.instance()


class Reducer:
    needs_position = False
    jax = import_jax()

    @classmethod
    def return_dtype(cls, given_dtype):
        if given_dtype in (np.bool_, np.int8, np.int16, np.int32):
            return np.int32 if ak._util.win or ak._util.bits32 else np.int64

        if given_dtype in (np.uint8, np.uint16, np.uint32):
            return np.uint32 if ak._util.win or ak._util.bits32 else np.uint64

        return given_dtype

    @classmethod
    def maybe_double_length(cls, type, length):
        return 2 * length if type in (np.complex128, np.complex64) else length

    @classmethod
    def maybe_other_type(cls, dtype):
        type = np.int64 if dtype.kind.upper() == "M" else dtype.type
        if dtype == np.complex128:
            type = np.float64
        if dtype == np.complex64:
            type = np.float32
        return type


class ArgMin(Reducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._v2._util.error("Cannot differentiate through argmin")


class ArgMax(Reducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._v2._util.error("Cannot differentiate through argmax")


class Count(Reducer):
    name = "count"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._v2._util.error("Cannot differentiate through count_zero")


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        raise ak._v2._util.error("Cannot differentiate through count_nonzero")


class Sum(Reducer):
    name = "sum"
    preferred_dtype = np.float64

    @classmethod
    def apply(cls, array, parents, outlength):

        assert isinstance(array, ak._v2.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ak._v2._util.error(
                ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")
            )

        result = cls.jax.ops.segment_sum(array.data, parents.data)

        if array.dtype.kind == "m":
            return ak._v2.contents.NumpyArray(array.nplike.asarray(result, array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak._v2.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak._v2.contents.NumpyArray(result, nplike=array.nplike)


class Prod(Reducer):
    name = "prod"
    preferred_dtype = np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak._v2.contents.NumpyArray)
        # See issue https://github.com/google/jax/issues/9296
        result = cls.jax.numpy.exp(
            cls.jax.ops.segment_sum(cls.jax.numpy.log(array.data), parents.data)
        )

        if array.dtype.type in (np.complex128, np.complex64):
            return ak._v2.contents.NumpyArray(
                result.view(array.dtype), nplike=array.nplike
            )
        else:
            return ak._v2.contents.NumpyArray(result, nplike=array.nplike)


class Any(Reducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak._v2.contents.NumpyArray)
        result = cls.jax.ops.segment_max(array.data, parents.data)
        result = cls.jax.numpy.asarray(result, dtype=bool)

        return ak._v2.contents.NumpyArray(result, nplike=array.nplike)


class All(Reducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        assert isinstance(array, ak._v2.contents.NumpyArray)
        result = cls.jax.ops.segment_min(array.data, parents.data)
        result = cls.jax.numpy.asarray(result, dtype=bool)

        return ak._v2.contents.NumpyArray(result, nplike=array.nplike)


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
        assert isinstance(array, ak._v2.contents.NumpyArray)

        result = cls.jax.ops.segment_min(array.data, parents.data)
        result = cls.jax.numpy.minimum(
            result, cls._min_initial(cls.initial, array.dtype)
        )

        if array.dtype.type in (np.complex128, np.complex64):
            return ak._v2.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype),
                nplike=array.nplike,
            )
        else:
            return ak._v2.contents.NumpyArray(result, nplike=array.nplike)


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
        assert isinstance(array, ak._v2.contents.NumpyArray)

        result = cls.jax.ops.segment_max(array.data, parents.data)

        result = cls.jax.numpy.maximum(
            result, cls._max_initial(cls.initial, array.dtype)
        )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak._v2.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype),
                nplike=array.nplike,
            )
        else:
            return ak._v2.contents.NumpyArray(result, nplike=array.nplike)
