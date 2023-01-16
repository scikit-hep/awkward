# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any as AnyType

import awkward as ak
from awkward._nplikes import metadata
from awkward.typing import Final

numpy = ak._nplikes.Numpy.instance()

DTypeLike = AnyType


class Reducer(ABC):
    name: str

    # Does the output correspond to array positions?
    @property
    @abstractmethod
    def needs_position(self) -> bool:
        ...

    @property
    @abstractmethod
    def preferred_dtype(self) -> DTypeLike:
        ...

    @classmethod
    def highlevel_function(cls):
        return getattr(ak.operations, cls.name)

    @classmethod
    def return_dtype(cls, given_dtype: DTypeLike) -> DTypeLike:
        if metadata.isdtype(
            given_dtype, (metadata.bool_, metadata.int8, metadata.int16, metadata.int32)
        ):
            return metadata.int32 if ak._util.is_32_bit() else metadata.int64

        if metadata.isdtype(
            given_dtype, (metadata.uint8, metadata.uint16, metadata.uint32)
        ):
            return metadata.uint32 if ak._util.is_32_bit() else metadata.uint64

        return given_dtype

    @classmethod
    def maybe_double_length(cls, type: DTypeLike, length: int) -> int:
        return 2 * length if metadata.isdtype(type, "complex floating") else length

    @classmethod
    def maybe_other_type(cls, dtype: DTypeLike) -> DTypeLike:
        assert isinstance(dtype, metadata.dtype)
        type = metadata.int64 if metadata.isdtype(dtype, "timelike") else dtype.type
        if dtype == metadata.complex128:
            type = metadata.float64
        if dtype == metadata.complex64:
            type = metadata.float32
        return type

    @abstractmethod
    def identity_for(self, dtype: DTypeLike | None):
        raise ak._errors.wrap_error(NotImplementedError)

    @abstractmethod
    def apply(self, array, parents, outlength: int):
        raise ak._errors.wrap_error(NotImplementedError)


class ArgMin(Reducer):
    name: Final = "argmin"
    needs_position: Final = True
    preferred_dtype: Final = metadata.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    def identity_for(self, dtype: metadata.dtype | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None
        return metadata.int64(-1)

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(outlength, dtype=metadata.int64)
        if metadata.isdtype(array.dtype, "complex floating"):
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


class ArgMax(Reducer):
    name: Final = "argmax"
    needs_position: Final = True
    preferred_dtype: Final = metadata.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    def identity_for(self, dtype: metadata.dtype | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None
        return metadata.int64(-1)

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(outlength, dtype=metadata.int64)
        if metadata.isdtype(array.dtype, "complex floating"):
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


class Count(Reducer):
    name: Final = "count"
    preferred_dtype: Final = metadata.int64
    needs_position: Final = False

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.backend.nplike.empty(outlength, dtype=metadata.int64)
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

    def identity_for(self, dtype: metadata.dtype | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None
        return metadata.int64(0)


class CountNonzero(Reducer):
    name: Final = "count_nonzero"
    preferred_dtype: Final = metadata.float64
    needs_position: Final = False

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = (
            metadata.int64 if metadata.isdtype(array.dtype, "timelike") else array.dtype
        )
        result = array.backend.nplike.empty(outlength, dtype=metadata.int64)
        if metadata.isdtype(array.dtype, "complex floating"):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
                    "awkward_reduce_countnonzero_complex",
                    result.dtype.type,
                    metadata.float64
                    if array.dtype.type == metadata.complex128
                    else metadata.float32,
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

    def identity_for(self, dtype: metadata.dtype | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None
        return metadata.int64(0)


class Sum(Reducer):
    name: Final = "sum"
    preferred_dtype: Final = metadata.float64
    needs_position: Final = False

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

        if metadata.isdtype(array.dtype, "bool"):
            if metadata.isdtype(result.dtype, (metadata.int64, metadata.uint64)):
                assert parents.nplike is array.backend.index_nplike
                array._handle_error(
                    array.backend[
                        "awkward_reduce_sum_int64_bool_64",
                        metadata.int64,
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
            elif metadata.isdtype(result.dtype, (metadata.int32, metadata.uint32)):
                assert parents.nplike is array.backend.index_nplike
                array._handle_error(
                    array.backend[
                        "awkward_reduce_sum_int32_bool_64",
                        metadata.int32,
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
        elif metadata.isdtype(array.dtype, "complex floating"):
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
                    "awkward_reduce_sum_complex",
                    metadata.float64
                    if array.dtype.type == metadata.complex128
                    else metadata.float32,
                    metadata.float64
                    if array.dtype.type == metadata.complex128
                    else metadata.float32,
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
                    metadata.int64 if array.dtype.kind == "m" else array.dtype.type,
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
                array.backend.nplike.asarray(result, dtype=array.dtype)
            )
        elif metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)

    def identity_for(self, dtype: metadata.dtype | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None
        if dtype is None:
            dtype = self.preferred_dtype

        if metadata.isdtype(dtype, "timelike"):
            return metadata.timedelta64(0)
        else:
            return numpy.asarray(0, dtype=dtype)[()]


class Prod(Reducer):
    name: Final = "prod"
    preferred_dtype: Final = metadata.int64
    needs_position: Final = False

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind.upper() == "M":
            raise ak._errors.wrap_error(
                ValueError(f"cannot compute the product (ak.prod) of {array.dtype!r}")
            )
        if metadata.isdtype(array.dtype, "bool"):
            result = array.backend.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=metadata.bool_,
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
            result = array.backend.nplike.astype(result, self.return_dtype(array.dtype))
        elif metadata.isdtype(array.dtype, "complex floating"):
            result = array.backend.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.backend.index_nplike
            array._handle_error(
                array.backend[
                    "awkward_reduce_prod_complex",
                    metadata.float64
                    if array.dtype.type == metadata.complex128
                    else metadata.float32,
                    metadata.float64
                    if array.dtype.type == metadata.complex128
                    else metadata.float32,
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
        if metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)

    def identity_for(self, dtype: metadata.dtype | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None
        if dtype is None:
            dtype = self.preferred_dtype

        if metadata.isdtype(dtype, "timelike"):
            return metadata.timedelta64(0)
        else:
            return numpy.asarray(1, dtype=dtype)[()]


class Any(Reducer):
    name: Final = "any"
    preferred_dtype: Final = metadata.bool_
    needs_position: Final = False

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(outlength, dtype=metadata.bool_)
        if metadata.isdtype(array.dtype, "complex floating"):
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

    def identity_for(self, dtype: DTypeLike | None) -> float:
        assert isinstance(dtype, metadata.dtype) or dtype is None
        return False


class All(Reducer):
    name: Final = "all"
    preferred_dtype: Final = metadata.bool_
    needs_position: Final = False

    @classmethod
    def return_dtype(cls, given_dtype):
        return metadata.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(outlength, dtype=metadata.bool_)
        if metadata.isdtype(array.dtype, "complex floating"):
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

    def identity_for(self, dtype: DTypeLike | None) -> float:
        assert isinstance(dtype, metadata.dtype) or dtype is None
        return True


class Min(Reducer):
    name: Final = "min"
    preferred_dtype: Final = metadata.float64
    needs_position: Final = False

    def __init__(self, initial: float | None):
        self._initial = initial

    @property
    def initial(self) -> float | None:
        return self._initial

    def identity_for(self, dtype: DTypeLike | None) -> float:
        assert isinstance(dtype, metadata.dtype) or dtype is None

        assert not metadata.isdtype(
            dtype, "timelike"
        ), "datetime64/timedelta64 should be converted to int64 before reduction"
        if self._initial is None:
            if metadata.isdtype(dtype, "integral"):
                return metadata.iinfo(dtype).max
            else:
                return metadata.inf

        return self._initial

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if metadata.isdtype(array.dtype, "bool"):
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
        elif metadata.isdtype(array.dtype, "complex floating"):
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
        if metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(
                    result.view(array.dtype), dtype=array.dtype
                )
            )
        else:
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(result, dtype=array.dtype)
            )


class Max(Reducer):
    name: Final = "max"
    preferred_dtype: Final = metadata.float64
    needs_position: Final = False

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    def identity_for(self, dtype: DTypeLike | None):
        assert isinstance(dtype, metadata.dtype) or dtype is None

        assert not metadata.isdtype(
            dtype, "timelike"
        ), "datetime64/timedelta64 should be converted to int64 before reduction"
        if self._initial is None:
            if metadata.isdtype(dtype, "integral"):
                return metadata.iinfo(dtype).min
            else:
                return -metadata.inf

        return self._initial

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.backend.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if metadata.isdtype(array.dtype, "bool"):
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
        elif metadata.isdtype(array.dtype, "complex floating"):
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
        if metadata.isdtype(array.dtype, "complex floating"):
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(
                    result.view(array.dtype), dtype=array.dtype
                )
            )
        else:
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(result, dtype=array.dtype)
            )
