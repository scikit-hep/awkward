# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from functools import reduce
from numbers import Integral, Real
from typing import Any

import awkward as ak
from awkward import nplikes

np = nplikes.NumpyMetadata.instance()
numpy = nplikes.Numpy.instance()


DTypeLike = Any


class Reducer:
    needs_position = False

    @classmethod
    def highlevel_function(cls):
        return getattr(ak.operations, cls.name)

    @classmethod
    def return_dtype(cls, given_dtype: DTypeLike) -> DTypeLike:
        if given_dtype in (np.bool_, np.int8, np.int16, np.int32):
            return np.int32 if ak._util.win or ak._util.bits32 else np.int64

        if given_dtype in (np.uint8, np.uint16, np.uint32):
            return np.uint32 if ak._util.win or ak._util.bits32 else np.uint64

        return given_dtype

    @classmethod
    def maybe_double_length(cls, type: DTypeLike, length: Integral) -> Integral:
        return 2 * length if type in (np.complex128, np.complex64) else length

    @classmethod
    def maybe_other_type(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        type = np.int64 if dtype.kind.upper() == "M" else dtype.type
        if dtype == np.complex128:
            type = np.float64
        if dtype == np.complex64:
            type = np.float32
        return type

    def identity_for(self, dtype: DTypeLike | None):
        raise ak._errors.wrap_error(NotImplementedError)

    def apply(self, array, parents, outlength: Integral):
        raise ak._errors.wrap_error(NotImplementedError)

    def apply_parts(self, arrays):
        raise ak._errors.wrap_error(NotImplementedError)


class ArgMin(Reducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmin_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmin",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        if len(arrays) == 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot compute the argmin (ak.argmin) of an array with empty leaves"
                )
            )
        else:
            nplike = ak.nplikes.nplike_of(*arrays)

            offset = 0
            best_index = None
            best_value = None
            for array in arrays:
                local_index = nplike.argmin(array, axis=None)
                value = array[local_index]
                if best_index is None or value < best_value:
                    best_value = value
                    best_index = local_index + offset
                offset += len(array)

            return best_index


class ArgMax(Reducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmax_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmax",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        if len(arrays) == 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot compute the argmax (ak.argmax) of an array with empty leaves"
                )
            )
        else:
            nplike = ak.nplikes.nplike_of(*arrays)

            offset = 0
            best_index = None
            best_value = None
            for array in arrays:
                local_index = nplike.argmax(array, axis=None)
                value = array[local_index]
                if best_index is None or value > best_value:
                    best_value = value
                    best_index = local_index + offset
                offset += len(array)

            return best_index


class Count(Reducer):
    name = "count"
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.nplike.empty(outlength, dtype=np.int64)
        assert parents.nplike is array.nplike
        array._handle_error(
            array.nplike[
                "awkward_reduce_count_64", result.dtype.type, parents.dtype.type
            ](
                result,
                parents.data,
                parents.length,
                outlength,
            )
        )
        return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.size(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.add, partials, self.identity_for(dtype))

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(0)


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = np.dtype(np.int64) if array.dtype.kind.upper() == "M" else array.dtype
        result = array.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_countnonzero_complex",
                    result.dtype.type,
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_countnonzero",
                    result.dtype.type,
                    dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.count_nonzero(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.add, partials, self.identity_for(dtype))

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(0)


class Sum(Reducer):
    name = "sum"
    preferred_dtype = np.float64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ak._errors.wrap_error(
                ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")
            )
        else:
            dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength),
            dtype=self.return_dtype(dtype),
        )

        if array.dtype == np.bool_:
            if result.dtype in (np.int64, np.uint64):
                assert parents.nplike is array.nplike
                array._handle_error(
                    array.nplike[
                        "awkward_reduce_sum_int64_bool_64",
                        np.int64,
                        array.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        array.data,
                        parents.data,
                        parents.length,
                        outlength,
                    )
                )
            elif result.dtype in (np.int32, np.uint32):
                assert parents.nplike is array.nplike
                array._handle_error(
                    array.nplike[
                        "awkward_reduce_sum_int32_bool_64",
                        np.int32,
                        array.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        array.data,
                        parents.data,
                        parents.length,
                        outlength,
                    )
                )
            else:
                raise ak._errors.wrap_error(NotImplementedError)
        elif array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_complex",
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum",
                    result.dtype.type,
                    np.int64 if array.dtype.kind == "m" else array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )

        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(array.nplike.asarray(result, array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.sum(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.add, partials, self.identity_for(dtype))

    def identity_for(self, dtype: np.dtype | None):
        if dtype is None:
            dtype = self.preferred_dtype

        if dtype in {np.timedelta64, np.datetime64}:
            return np.timedelta64(0)
        else:
            return numpy.array(0, dtype=dtype)[()]


class Prod(Reducer):
    name = "prod"
    preferred_dtype = np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind.upper() == "M":
            raise ak._errors.wrap_error(
                ValueError(f"cannot compute the product (ak.prod) of {array.dtype!r}")
            )
        if array.dtype == np.bool_:
            result = array.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=np.dtype(np.bool_),
            )
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
            result = result.astype(self.return_dtype(array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            result = array.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_complex",
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            result = array.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.prod(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.multiply, partials, self.identity_for(dtype))

    def identity_for(self, dtype: np.dtype | None):
        if dtype is None:
            dtype = self.preferred_dtype

        if dtype in {np.timedelta64, np.datetime64}:
            return np.timedelta64(0)
        else:
            return numpy.array(1, dtype=dtype)[()]


class Any(Reducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.bool_)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.any(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.logical_or, partials, self.identity_for(dtype))

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        return False


class All(Reducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.bool_)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.all(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.logical_and, partials, self.identity_for(dtype))

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        return True


class Min(Reducer):
    name = "min"
    preferred_dtype = np.float64

    def __init__(self, initial: Real | None):
        self._initial = initial

    @property
    def initial(self) -> Real | None:
        return self._initial

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
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if array.dtype == np.bool_:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        elif array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_min_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self.identity_for(dtype),
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_min",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self.identity_for(dtype),
                )
            )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype)
            )
        else:
            return ak.contents.NumpyArray(array.nplike.array(result, array.dtype))

    def apply_parts(self, arrays):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.min(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.minimum, partials, self.identity_for(dtype))


class Max(Reducer):
    name = "max"
    preferred_dtype = np.float64

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

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
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if array.dtype == np.bool_:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        elif array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_max_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self.identity_for(dtype),
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_max",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self.identity_for(dtype),
                )
            )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype)
            )
        else:
            return ak.contents.NumpyArray(array.nplike.array(result, array.dtype))

    def apply_parts(self, arrays: list):
        nplike = ak.nplikes.nplike_of(*arrays)
        partials = [nplike.max(x, axis=None) for x in arrays]
        dtype = nplike.common_type([x.dtype for x in arrays])
        return reduce(nplike.maximum, partials, self.identity_for(dtype))
