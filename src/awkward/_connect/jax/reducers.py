# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

import jax

import awkward as ak
from awkward import _reducers
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._nplikes.virtual import materialize_if_virtual
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


def awkward_JAXNumpyArray_reduce_adjust_starts_64(toptr, outlength, parents, starts):
    if outlength == 0 or parents.size == 0:
        return toptr

    identity = jax.numpy.iinfo(jax.numpy.int64).max
    valid = toptr[:outlength] != identity
    safe_sub_toptr = jax.numpy.where(valid, toptr[:outlength], 0)
    parent_indices = parents[safe_sub_toptr]
    adjustments = starts[parent_indices]
    updated = jax.numpy.where(valid, toptr[:outlength] - adjustments, toptr[:outlength])

    return toptr.at[:outlength].set(updated)


def awkward_JAXNumpyArray_reduce_adjust_starts_shifts_64(
    toptr, outlength, parents, starts, shifts
):
    if outlength == 0 or parents.size == 0:
        return toptr

    identity = jax.numpy.iinfo(jax.numpy.int64).max
    valid = toptr[:outlength] != identity
    safe_sub_toptr = jax.numpy.where(valid, toptr[:outlength], 0)
    parent_indices = parents[safe_sub_toptr]
    delta = shifts[safe_sub_toptr] - starts[parent_indices]
    updated = jax.numpy.where(valid, toptr[:outlength] + delta, toptr[:outlength])

    return toptr.at[:outlength].set(updated)


def apply_positional_corrections(
    reduced: ak.contents.NumpyArray,
    parents: ak.index.Index,
    starts: ak.index.Index,
    shifts: ak.index.Index | None,
) -> ak._nplikes.ArrayLike:
    if shifts is None:
        assert (
            parents.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
        )
        return awkward_JAXNumpyArray_reduce_adjust_starts_64(
            reduced.data, reduced.length, parents.data, starts.data
        )

    else:
        assert (
            parents.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
            and shifts.nplike is reduced.backend.nplike
        )
        return awkward_JAXNumpyArray_reduce_adjust_starts_shifts_64(
            reduced.data,
            reduced.length,
            parents.data,
            starts.data,
            shifts.data,
        )


def segment_argmin(data, segment_ids, num_segments):
    """
    Applies a segmented argmin-style reduction.

    Parameters:
        data: jax.numpy.ndarray — the values to reduce.
        segment_ids: same shape as data — indicates segment groupings.
        num_segments: int — total number of segments.

    Returns:
        jax.numpy.ndarray — indices of min within each segment.
    """
    indices = jax.numpy.arange(data.shape[0])

    # Find the minimum value in each segment
    min_vals = jax.ops.segment_min(data, segment_ids, num_segments=num_segments)

    # Find where the data equals the minimum value in each segment
    is_min = data == min_vals[segment_ids]

    # Mask the indices where data matches the minimum value
    masked_indices = jax.numpy.where(is_min, indices, data.shape[0])

    # Return the index of the first minimum value in each segment
    return jax.ops.segment_min(masked_indices, segment_ids, num_segments=num_segments)


@overloads(_reducers.ArgMin)
class ArgMin(JAXReducer):
    name: Final = "argmin"
    needs_position: Final = True
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.ArgMin)
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
        result = segment_argmin(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
        result = jax.numpy.asarray(result, dtype=self.preferred_dtype)
        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        corrected_data = apply_positional_corrections(
            result_array, parents, starts, shifts
        )
        return ak.contents.NumpyArray(corrected_data, backend=array.backend)


def segment_argmax(data, segment_ids, num_segments):
    """
    Applies a segmented argmax-style reduction.

    Parameters:
        data: jax.numpy.ndarray — the values to reduce.
        segment_ids: same shape as data — indicates segment groupings.
        num_segments: int — total number of segments.

    Returns:
        jax.numpy.ndarray — indices of max within each segment.
    """
    indices = jax.numpy.arange(data.shape[0])

    # Find the maximum value in each segment
    max_vals = jax.ops.segment_max(data, segment_ids, num_segments=num_segments)

    # Find where the data equals the maximum value in each segment
    is_max = data == max_vals[segment_ids]

    # Mask the indices where data matches the maximum value
    masked_indices = jax.numpy.where(is_max, indices, data.shape[0])

    # Return the index of the first maximum value in each segment
    return jax.ops.segment_min(masked_indices, segment_ids, num_segments=num_segments)


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
        assert isinstance(array, ak.contents.NumpyArray)
        result = segment_argmax(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
        result = jax.numpy.asarray(result, dtype=self.preferred_dtype)
        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        corrected_data = apply_positional_corrections(
            result_array, parents, starts, shifts
        )
        return ak.contents.NumpyArray(corrected_data, backend=array.backend)


@overloads(_reducers.Count)
class Count(JAXReducer):
    name: Final = "count"
    preferred_dtype: Final = np.int64
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
        result = jax.numpy.ones_like(
            *materialize_if_virtual(array.data), dtype=self.preferred_dtype
        )
        result = jax.ops.segment_sum(
            result, *materialize_if_virtual(parents.data), outlength
        )
        result = jax.numpy.asarray(result, dtype=self.preferred_dtype)

        if np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


def segment_count_nonzero(data, segment_ids, num_segments):
    """
    Counts the number of non-zero elements in `data` per segment.

    Parameters:
        data: jax.numpy.ndarray — input values to count.
        segment_ids: jax.numpy.ndarray — same shape as data, segment assignment.
        num_segments: int — total number of segments.

    Returns:
        jax.numpy.ndarray — count of non-zero values per segment.
    """
    # Create a binary mask where non-zero entries become 1
    nonzero_mask = jax.numpy.where(data != 0, 1, 0)

    # Sum the mask using segment_sum to count per segment
    return jax.ops.segment_sum(nonzero_mask, segment_ids, num_segments=num_segments)


@overloads(_reducers.CountNonzero)
class CountNonzero(JAXReducer):
    name: Final = "count_nonzero"
    preferred_dtype: Final = np.int64
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
        assert isinstance(array, ak.contents.NumpyArray)
        result = segment_count_nonzero(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
        result = jax.numpy.asarray(result, dtype=self.preferred_dtype)

        return ak.contents.NumpyArray(result, backend=array.backend)


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

        if array.dtype.kind == "b":
            input_array = array.data.astype(np.int64)
        else:
            input_array = array.data
        result = jax.ops.segment_sum(
            *materialize_if_virtual(input_array, parents.data), outlength
        )

        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(
                array.backend.nplike.asarray(result, dtype=array.dtype)
            )
        elif np.issubdtype(array.dtype, np.complexfloating):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result, backend=array.backend)


def segment_prod_with_negatives(data, segment_ids, num_segments):
    """
    Computes the product of elements in each segment, handling negatives and booleans.
    Parameters:
        data: jax.numpy.ndarray — input values to reduce.
        segment_ids: jax.numpy.ndarray — same shape as data, segment assignment.
        num_segments: int — total number of segments.
    Returns:
        jax.numpy.ndarray — product of values per segment.
    """
    # Handle boolean arrays
    if data.dtype == jax.numpy.bool_:
        return jax.ops.segment_min(
            data.astype(jax.numpy.int32), segment_ids, num_segments
        )

    # For numeric arrays, handle negative values and zeros
    # Track zeros to set product to zero if any segment has zeros
    is_zero = data == 0
    has_zeros = (
        jax.ops.segment_sum(is_zero.astype(jax.numpy.int32), segment_ids, num_segments)
        > 0
    )

    # Track signs to determine final sign of product
    is_negative = data < 0
    neg_count = jax.ops.segment_sum(
        is_negative.astype(jax.numpy.int32), segment_ids, num_segments
    )
    sign_products = 1 - 2 * (
        neg_count % 2
    )  # +1 for even negatives, -1 for odd negatives

    # Calculate product of absolute values in log space
    log_abs = jax.numpy.log(jax.numpy.where(is_zero, 1.0, jax.numpy.abs(data)))
    log_products = jax.ops.segment_sum(
        jax.numpy.where(is_zero, 0.0, log_abs), segment_ids, num_segments
    )
    abs_products = jax.numpy.exp(log_products)

    # Apply zeros and signs
    product = jax.numpy.where(has_zeros, 0.0, sign_products * abs_products)
    # floating point accuracy doesn't let us directly cast to integers
    if np.issubdtype(data.dtype, np.integer):
        result = jax.numpy.round(product).astype(data.dtype)
    else:
        result = product
    return result


@overloads(_reducers.Prod)
class Prod(JAXReducer):
    name: Final = "prod"
    preferred_dtype: Final = np.float64
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
        result = segment_prod_with_negatives(
            *materialize_if_virtual(array.data, parents.data), outlength
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
        result = jax.ops.segment_max(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
        if array.dtype is not np.dtype(bool):
            result = result.at[result == 0].set(self._max_initial(None, array.dtype))
            result = result > self._max_initial(None, array.dtype)
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
        result = jax.ops.segment_min(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
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

        result = jax.ops.segment_min(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
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

        result = jax.ops.segment_max(
            *materialize_if_virtual(array.data, parents.data), outlength
        )
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
