# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()

DTypeLike = Any


class Reducer:
    name: str

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

    def _apply_many_trivial(self, arrays, mask, partial_reducer=None):
        if partial_reducer is None:
            partial_reducer = self

        backend = ak._backends.backend_of(*arrays)
        partial_reductions = [
            x._reduce_next(
                self,
                negaxis=1,
                starts=ak.index.Index64.zeros(
                    1, backend.index_nplike
                ),  # TODO: add to result for positional?
                parents=ak.index.Index64.zeros(x.length, backend.index_nplike),
                shifts=None,
                mask=mask,
                outlength=1,
                keepdims=True,
                behavior=None,
            )[0]
            for x in arrays
        ]
        partial_reduction = partial_reductions[0]._mergemany(partial_reductions[1:])
        return partial_reduction._reduce_next(
            partial_reducer,
            negaxis=1,
            mask=mask,
            keepdims=True,
            starts=ak.index.Index64.zeros(1, backend.index_nplike),
            parents=ak.index.Index64.zeros(
                partial_reduction.length, backend.index_nplike
            ),
            shifts=None,
            outlength=1,
            behavior=None,
        )[0]

    def apply_many(self, arrays: list[ak.contents.Content], mask: bool):
        """
        Args:
            arrays: arrays to be partially reduced

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
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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

    def apply_many(self, arrays, mask):
        backend = ak._backends.backend_of(*arrays)

        if backend.nplike.known_shape and backend.nplike.known_data:
            partial_indices = [
                part._reduce_next(
                    self,
                    negaxis=1,
                    starts=ak.index.Index64.zeros(1, backend.index_nplike),
                    parents=ak.index.Index64.zeros(part.length, backend.index_nplike),
                    shifts=None,
                    mask=True,
                    outlength=1,
                    keepdims=True,
                    behavior=None,
                )[0]
                for part in arrays
            ]
            index = (
                partial_indices[0]
                ._mergemany(partial_indices[1:])
                .to_ByteMaskedArray(True)
            )

            partial_values = [a[i] for a, i in zip(arrays, partial_indices)]
            value = partial_values[0]._mergemany(partial_values[1:])

            result_index = value._reduce_next(
                self,
                negaxis=1,
                mask=True,
                keepdims=True,
                starts=ak.index.Index64.zeros(1, backend.index_nplike),
                parents=ak.index.Index64.zeros(value.length, backend.index_nplike),
                shifts=None,
                outlength=1,
                behavior=None,
            )[0]
            assert isinstance(result_index, ak.contents.ByteMaskedArray), type(value)

            lengths = [0]
            lengths.extend([len(x) for x in arrays[:-1]])
            shifts = backend.index_nplike.where(
                index.mask,
                backend.index_nplike.cumsum(
                    backend.index_nplike.array(lengths, dtype=np.int64)
                ),
                0,
            )
            absolute_index = ak.contents.NumpyArray(
                backend.index_nplike.asarray(index.content) + shifts
            )

            result = absolute_index[result_index]
            if mask:
                return result
            else:
                raw_result = result.to_ByteMaskedArray(True)
                return ak.contents.NumpyArray(
                    backend.index_nplike.where(
                        raw_result.mask,
                        backend.index_nplike.asarray(raw_result.content),
                        -1,
                    )
                )
        else:
            content = ak.contents.NumpyArray(
                backend.nplike.empty(1, dtype=np.int64), backend=backend
            )
            if mask:
                return ak.contents.ByteMaskedArray(
                    ak.index.Index8(backend.index_nplike.empty(1, dtype=np.int8)),
                    content,
                    True,
                )
            else:
                return content


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
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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

    def apply_many(self, arrays, mask):
        backend = ak._backends.backend_of(*arrays)

        if backend.nplike.known_shape and backend.nplike.known_data:
            partial_indices = [
                part._reduce_next(
                    self,
                    negaxis=1,
                    starts=ak.index.Index64.zeros(1, backend.index_nplike),
                    parents=ak.index.Index64.zeros(part.length, backend.index_nplike),
                    shifts=None,
                    mask=True,
                    outlength=1,
                    keepdims=True,
                    behavior=None,
                )[0]
                for part in arrays
            ]
            index = (
                partial_indices[0]
                ._mergemany(partial_indices[1:])
                .to_ByteMaskedArray(True)
            )

            partial_values = [a[i] for a, i in zip(arrays, partial_indices)]
            value = partial_values[0]._mergemany(partial_values[1:])

            result_index = value._reduce_next(
                self,
                negaxis=1,
                mask=True,
                keepdims=True,
                starts=ak.index.Index64.zeros(1, backend.index_nplike),
                parents=ak.index.Index64.zeros(value.length, backend.index_nplike),
                shifts=None,
                outlength=1,
                behavior=None,
            )[0]
            assert isinstance(result_index, ak.contents.ByteMaskedArray), type(value)

            lengths = [0]
            lengths.extend([len(x) for x in arrays[:-1]])
            shifts = backend.index_nplike.where(
                index.mask,
                backend.index_nplike.cumsum(
                    backend.index_nplike.array(lengths, dtype=np.int64)
                ),
                0,
            )
            absolute_index = ak.contents.NumpyArray(
                backend.index_nplike.asarray(index.content) + shifts
            )

            result = absolute_index[result_index]
            if mask:
                return result
            else:
                raw_result = result.to_ByteMaskedArray(True)
                return ak.contents.NumpyArray(
                    backend.index_nplike.where(
                        raw_result.mask,
                        backend.index_nplike.asarray(raw_result.content),
                        -1,
                    )
                )
        else:
            content = ak.contents.NumpyArray(
                backend.nplike.empty(1, dtype=np.int64), backend=backend
            )
            if mask:
                return ak.contents.ByteMaskedArray(
                    ak.index.Index8(backend.index_nplike.empty(1, dtype=np.int8)),
                    content,
                    True,
                )
            else:
                return content


class Count(Reducer):
    name = "count"
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        assert parents.nplike is array.backend.index_nplike
        array._handle_error(
            array.backend[
                "awkward_reduce_count_64", result.dtype.type, parents.dtype.type
            ](
                result,
                parents.data,
                parents.length,
                outlength,
            )
        )
        return ak.contents.NumpyArray(result)

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(0)

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask, partial_reducer=Sum())


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = np.dtype(np.int64) if array.dtype.kind.upper() == "M" else array.dtype
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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

    def identity_for(self, dtype: np.dtype | None):
        return np.int64(0)

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask, partial_reducer=Sum())


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
        result = array.backend.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength),
            dtype=self.return_dtype(dtype),
        )

        if array.dtype == np.bool_:
            if result.dtype in (np.int64, np.uint64):
                assert parents.nplike is array.backend.index_nplike
                array._handle_error(
                    array.backend[
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
                assert parents.nplike is array.backend.index_nplike
                array._handle_error(
                    array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(result, array.dtype)
            )
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)

    def identity_for(self, dtype: np.dtype | None):
        if dtype is None:
            dtype = self.preferred_dtype

        if dtype in {np.timedelta64, np.datetime64}:
            return np.timedelta64(0)
        else:
            return numpy.array(0, dtype=dtype)[()]

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask)


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
            result = array.backend.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=np.dtype(np.bool_),
            )
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            result = array.backend.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            result = array.backend.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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

    def identity_for(self, dtype: np.dtype | None):
        if dtype is None:
            dtype = self.preferred_dtype

        if dtype in {np.timedelta64, np.datetime64}:
            return np.timedelta64(0)
        else:
            return numpy.array(1, dtype=dtype)[()]

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask)


class Any(Reducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        return False

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask)


class All(Reducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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

    def identity_for(self, dtype: DTypeLike | None) -> Real:
        return True

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask)


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
        result = array.backend.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if array.dtype == np.bool_:
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
                array.backend.nplike.array(result.view(array.dtype), array.dtype)
            )
        else:
            return ak.contents.NumpyArray(
                array.backend.nplike.array(result, array.dtype)
            )

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask)


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
        result = array.backend.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if array.dtype == np.bool_:
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
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
                array.backend.nplike.array(result.view(array.dtype), array.dtype)
            )
        else:
            return ak.contents.NumpyArray(
                array.backend.nplike.array(result, array.dtype)
            )

    def apply_many(self, arrays, mask):
        return self._apply_many_trivial(arrays, mask)
