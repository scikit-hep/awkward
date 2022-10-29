# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import awkward as ak
from awkward import nplikes

np = nplikes.NumpyMetadata.instance()
numpy = nplikes.Numpy.instance()


DTypeLike = Any


class Reducer:
    # Does the output correspond to array positions?
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

    def _combine_with(self, reductions, mask, reducer=None):
        if reducer is None:
            reducer = self

        partial = ak.contents.UnionArray(
            ak.index.Index8(
                self._nplike.index_nplike.arange(len(reductions), dtype=np.int8)
            ),
            ak.index.Index64(
                self._nplike.index_nplike.zeros(len(reductions), dtype=np.int64)
            ),
            reductions,
        )
        return partial.reduce(
            self, axis=-1, mask=mask, keepdims=False, flatten_records=False
        )

    def combine(
        self,
        array: ak.contents.Content,
        reduction,
        other_reduction,
        offset: Integral,
        aux,
    ):
        """
        Args:
            array: array to be partially reduced
            reduction: result of reducing `array`
            other_reduction: previous reduction result
            offset: position of partial reduction in the array formed by merging all partial arrays
            aux: auxiliary data returned by a previous call to `combine`

        Combine a pair of partial reductions such that the result is equivalent to the reduction over
        the array formed by merging the input arrays.
        """
        raise ak._errors.wrap_error(NotImplementedError)


class ArgMin(Reducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(-1)

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

    def combine(
        self,
        array: ak.contents.Content,
        reduction: Integral,
        other_reduction: Integral | None,
        offset: Integral,
        aux: Any,
    ) -> tuple[int, Any]:
        # Only slice array for valid indices
        if reduction == -1:
            this_index = -1
            this_value = None
        else:
            this_index = reduction + offset
            this_value = array[reduction]

        # If the previous reduction is not valid, just return this one
        if other_reduction is None or other_reduction == -1:
            return this_index, this_value
        # Else, if this reduction is smaller, return this result
        elif this_value < aux:
            return this_index, this_value
        # Else, return the previous reduction
        else:
            return other_reduction, aux

    def combine_many(self, partials: ak.contents.UnionArray, arrays, mask):
        nplike = ak.nplikes.nplike_of(*partials)
        index = ak.contents.UnionArray(
            ak.index.Index8(nplike.index_nplike.arange(len(partials), dtype=np.int8)),
            ak.index.Index64(nplike.index_nplike.zeros(len(partials), dtype=np.int64)),
            partials,
        )
        if not mask:
            index_array = nplike.index_nplike.asarray(
                index.simplify_uniontype(mergebool=True)
            )
            index_is_identity = nplike.index_nplike.equal(index_array, -1)
            bytemask = ak.index.Index8(index_is_identity.astype(np.int8))
            index = (
                ak.contents.ByteMaskedArray(
                    bytemask, index, valid_when=True
                ).simplify_optiontype(),
            )

        value = ak.contents.UnionArray(
            ak.index.Index8(nplike.index_nplike.arange(len(arrays), dtype=np.int8)),
            ak.index.Index64(nplike.index_nplike.zeros(len(arrays), dtype=np.int64)),
            [array[partial] for array, partial in zip(arrays, partials)],
        )

        local_result = value.reduce(
            self, axis=-1, mask=mask, keepdims=True, flatten_records=False
        )
        offsets = [0] * len(arrays)
        for i in range(1, len(arrays)):
            offsets[i] = len(arrays[i - 1]) + offsets[i - 1]
        offsets = ak.contents.NumpyArray(index.nplike.array(offsets))
        import numpy

        result = numpy.add(ak.Array(index), ak.Array(offsets))[local_result].layout

        if not mask:
            result = result.fill_none(-1)
        return result


class ArgMax(Reducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(-1)

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

    def combine(
        self,
        array: ak.contents.Content,
        reduction: Integral,
        other_reduction: Integral | None,
        offset: Integral,
        aux: Any,
    ) -> tuple[int, Any]:
        # Only slice array for valid indices
        if reduction == -1:
            this_index = -1
            this_value = None
        else:
            this_index = reduction + offset
            this_value = array[reduction]

        # If the previous reduction is not valid, just return this one
        if other_reduction is None or other_reduction == -1:
            return this_index, this_value
        # Else, if this reduction is smaller, return this result
        elif this_value > aux:
            return this_index, this_value
        # Else, return the previous reduction
        else:
            return other_reduction, aux

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        if not mask:
            bytemask = ak.index.Index8(
                partial.nplike.index_nplike.asarray(partial != -1).astype(np.int8)
            )
            partial = (
                ak.contents.ByteMaskedArray(
                    bytemask, partial, valid_when=True
                ).simplify_optiontype(),
            )

        value_content = ak.contents.UnionArray(
            ak.index.Index8(
                partial.nplike.index_nplike.arange(partial.length, dtype=np.int8)
            ),
            ak.index.Index64(
                partial.nplike.index_nplike.zeros(partial.length, dtype=np.int64)
            ),
            [arrays[i] for i in partial],
        )

        local_result = value_content.reduce(
            self, axis=-1, mask=mask, keepdims=False, flatten_records=False
        )
        offsets = [0] * len(arrays)
        for i in range(1, len(arrays)):
            offsets[i] = len(arrays[i - 1]) + offsets[i - 1]
        offsets = ak.contents.NumpyArray(partial.nplike.array(offsets))
        result = (partial + offsets)[local_result]

        if not mask:
            result = result.fill_none(-1)
        return result


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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return reduction + other_reduction, None

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(0)

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask, reducer=Sum())


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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return reduction + other_reduction, None

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(0)

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask, reducer=Sum())


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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return reduction + other_reduction, None

    def identity_for(self, dtype: np.dtype | None):
        if dtype is None:
            dtype = self.preferred_dtype

        if dtype in {np.timedelta64, np.datetime64}:
            return np.timedelta64(0)
        else:
            return numpy.array(0, dtype=dtype)[()]

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask)


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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return reduction * other_reduction, None

    def identity_for(self, dtype: np.dtype | None):
        if dtype is None:
            dtype = self.preferred_dtype

        if dtype in {np.timedelta64, np.datetime64}:
            return np.timedelta64(0)
        else:
            return numpy.array(1, dtype=dtype)[()]

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask)


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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return reduction or other_reduction, None

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        return False

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask)


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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return reduction and other_reduction, None

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        return True

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask)


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

        assert (
            dtype.kind.upper() != "M"
        ), "datetime64/timedelta64 should be converted to int64 before reduction"
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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return array.nplike.minimum(reduction, other_reduction), None

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask)


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

        assert (
            dtype.kind.upper() != "M"
        ), "datetime64/timedelta64 should be converted to int64 before reduction"
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

    def combine(self, array, reduction, other_reduction, offset, aux):
        if other_reduction is None:
            return reduction, None
        else:
            return array.nplike.maximum(reduction, other_reduction), None

    def combine_many(self, partial: ak.contents.UnionArray, arrays, mask):
        return self._combine_with(partial, mask)
