# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Reusable CCCL offset transforms, without owning caller buffers."""

from functools import lru_cache
from threading import RLock

import numpy as np
from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    make_inclusive_scan,
    make_reduce_into,
    make_segmented_reduce,
    make_unary_transform,
)
from cuda.compute.op import make_op_adapter

from awkward._nplikes.cupy import Cupy

cp = Cupy.instance()._module
# CCCL prepared transforms mutate their launch descriptors. Serialize host setup
# and launch, including factories that may share a globally cached descriptor.
_lock = RLock()


class _ArrayState:
    # CCCL rewrites captured CUDA-array-interface objects into device arrays.
    # These proxies carry launch metadata only; indexing in compiled operators
    # acts on the replacement arrays, not on this Python class.
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.clear()

    def bind(self, array):
        interface = array.__cuda_array_interface__
        self.__cuda_array_interface__ = {
            "shape": tuple(interface["shape"]),
            "strides": interface.get("strides"),
            "typestr": interface["typestr"],
            "data": (interface["data"][0], False),
            "version": 3,
        }

    def __len__(self):
        return self.__cuda_array_interface__["shape"][0]

    def data_ptr(self):
        return self.__cuda_array_interface__["data"][0]

    def clear(self):
        self.__cuda_array_interface__ = {
            "shape": (0,),
            "strides": None,
            "typestr": self.dtype.str,
            "data": (0, False),
            "version": 3,
        }


class _Gather:
    def __init__(self, outer_dtype, inner_dtype):
        self.outer = outer = _ArrayState(outer_dtype)
        self.inner = inner = _ArrayState(inner_dtype)

        def gather(i):
            return inner[outer[i]]

        self.operation = None
        self.gather = gather
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None

    def __call__(self, output, outer, inner, length, stream):
        self.outer.bind(outer)
        self.inner.bind(inner)
        try:
            self._launch(output, length, stream)
        finally:
            # Kernel arguments have been copied at launch. Retain no array
            # owners or active pointers in the Python operator's captured state.
            self.outer.clear()
            self.inner.clear()

    def _launch(self, output, length, stream):
        if self.operation is None:
            self.operation = make_op_adapter(self.gather)
        if self.prepared is None:
            self.prepared = make_unary_transform(
                d_in=self.iterator, d_out=output, op=self.operation
            )
        self.prepared(
            d_in=self.iterator,
            d_out=output,
            op=self.operation,
            num_items=length,
            stream=stream,
        )


@lru_cache(maxsize=16)
def _setup(device, context, outer_dtype, inner_dtype, output_dtype):
    return _Gather(outer_dtype, inner_dtype)


def gather_offsets(output, outer, inner, length, stream):
    if not length:
        return
    with _lock:
        setup = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            outer.dtype.str,
            inner.dtype.str,
            output.dtype.str,
        )
        setup(output, outer, inner, length, stream)


class _Ranges(_Gather):
    # Width is host metadata and part of the setup key. The backward search
    # avoids materializing one position per element or launching a reduction.
    def __init__(self, starts_dtype, source_dtype, width):
        self.outer = starts = _ArrayState(starts_dtype)
        self.inner = source = _ArrayState(source_dtype)

        def find_stop(i):
            start = i * width
            stop = start + width
            while stop > start and source[stop - 1] == -1:
                stop -= 1
            starts[i] = start  # pylint: disable=unsupported-assignment-operation
            return stop

        self.operation = None
        self.gather = find_stop
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None


@lru_cache(maxsize=16)
def _ranges_setup(
    device, context, starts_dtype, source_dtype, stops_dtype, width, groups
):
    # Bound serial scanning to short groups. Long groups need parallel work
    # even when their count is too small to fill the device with one thread each.
    if width > 64:
        return _ParallelRanges(starts_dtype, source_dtype, width, groups)
    return _Ranges(starts_dtype, source_dtype, width)


def nonlocal_offsets(starts, stops, source, length, groups, stream):
    width = length // groups
    with _lock:
        setup = _ranges_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            starts.dtype.str,
            source.dtype.str,
            stops.dtype.str,
            width,
            groups if width > 64 else 0,
        )
        setup(stops, starts, source, groups, stream)


def _first_error(left, right):
    # Zero is success; magnitude orders rows and sign preserves the error kind.
    # row + 1 fits int64 because a row precedes the final declared offset.
    if left == 0:
        return right
    if right == 0:
        return left
    return left if abs(left) <= abs(right) else right


class _Regular(_Gather):
    def __init__(self, offsets_dtype, rows, chunk_size):
        self.outer = size = _ArrayState(np.int64)
        self.inner = offsets = _ArrayState(offsets_dtype)

        def check(i):
            first = np.int64(offsets[1]) - np.int64(offsets[0])
            start = i * chunk_size
            stop = start + 1 if chunk_size == 1 else min(start + chunk_size, rows)
            if start == 0:
                # Assignment targets the device array substituted by CCCL.
                # pylint: disable-next=unsupported-assignment-operation
                size[0] = -1 if first < 0 else first
            for row in range(start, stop):
                count = np.int64(offsets[row + 1]) - np.int64(offsets[row])
                if count < 0:
                    return -(row + 1)
                if count != first:
                    return row + 1
            return np.int64(0)

        self.chunks = (rows + chunk_size - 1) // chunk_size
        self.chunk_size = chunk_size
        self.operation = None
        self.gather = check
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None
        self.reducer = None
        self.reduce_operation = None
        self.initial = np.asarray(0, dtype=np.int64)

    def validate(self, size, offsets, length, stream):
        chunks = length - 1 if self.chunk_size == 1 else self.chunks
        status = cp.empty(1, dtype=cp.int64)
        if chunks == 1:
            self(status, size, offsets, 1, stream)
            return status
        errors = cp.empty(chunks, dtype=cp.int64)
        self(errors, size, offsets, chunks, stream)
        if self.reducer is None:
            self.reduce_operation = make_op_adapter(_first_error)
            self.reducer = make_reduce_into(
                d_in=errors,
                d_out=status,
                op=self.reduce_operation,
                h_init=self.initial,
            )
        arguments = {
            "d_in": errors,
            "d_out": status,
            "num_items": chunks,
            "op": self.reduce_operation,
            "h_init": self.initial,
            "stream": stream,
        }
        storage_bytes = self.reducer(temp_storage=None, **arguments)
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        self.reducer(temp_storage=storage, **arguments)
        return status


@lru_cache(maxsize=16)
def _regular_setup(device, context, offsets_dtype, rows, chunk_size):
    return _Regular(offsets_dtype, rows, chunk_size)


def regular_status(size, offsets, length, stream, *, chunk_size=64):
    # The A100 sweep favors 64-row chunks: bounded serial work, fewer error
    # values, and no reduction launch for inputs containing at most 64 rows.
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with _lock:
        setup = _regular_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            offsets.dtype.str,
            0 if chunk_size == 1 else length - 1,
            chunk_size,
        )
        return setup.validate(size, offsets, length, stream)


class _ParallelRanges(_Gather):
    def __init__(self, starts_dtype, source_dtype, width, groups):
        self.outer = starts = _ArrayState(starts_dtype)
        self.inner = source = _ArrayState(source_dtype)
        self.boundaries = boundaries = _ArrayState(np.int64)

        def positions(i):
            if i < groups:
                # pylint: disable-next=unsupported-assignment-operation
                starts[i] = i * width
            if i <= groups:
                # pylint: disable-next=unsupported-assignment-operation
                boundaries[i] = i * width
            # Empty groups reduce to their own start, avoiding a clamp kernel.
            return i + 1 if source[i] != -1 else (i // width) * width

        self.operation = None
        self.gather = positions
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None
        self.reducer = None
        self.reduce_operation = None
        self.initial = np.asarray(0, dtype=np.int64)
        self.width = width

    def __call__(self, stops, starts, source, groups, stream):
        length = groups * self.width
        positions = cp.empty(length, dtype=cp.int64)
        boundaries = cp.empty(groups + 1, dtype=cp.int64)
        self.outer.bind(starts)
        self.inner.bind(source)
        self.boundaries.bind(boundaries)
        try:
            self._launch(positions, length, stream)
        finally:
            self.outer.clear()
            self.inner.clear()
            self.boundaries.clear()
        first = boundaries[:-1]
        last = boundaries[1:]
        if self.reducer is None:
            self.reduce_operation = make_op_adapter(OpKind.MAXIMUM)
            self.reducer = make_segmented_reduce(
                d_in=positions,
                d_out=stops,
                start_offsets_in=first,
                end_offsets_in=last,
                op=self.reduce_operation,
                h_init=self.initial,
            )
        arguments = {
            "d_in": positions,
            "d_out": stops,
            "num_segments": groups,
            "start_offsets_in": first,
            "end_offsets_in": last,
            "op": self.reduce_operation,
            "h_init": self.initial,
            "max_segment_size": self.width,
            "stream": stream,
        }
        storage_bytes = self.reducer(temp_storage=None, **arguments)
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        self.reducer(temp_storage=storage, **arguments)


class _DropCount(_Gather):
    def __init__(self, dtype, length_offsets, length_indexes):
        self.outer = offsets = _ArrayState(dtype)
        self.inner = indexes = _ArrayState(dtype)

        def count(i):
            if offsets[length_offsets - 1] > length_indexes:
                return np.int64(-1) if i == 0 else np.int64(0)
            total = np.int64(0)
            j = np.int64(0) if i == 0 else np.int64(offsets[i - 1])
            while j < np.int64(offsets[i]):
                if indexes[j] < 0:
                    total += 1
                j += 1
            return total

        self.operation = None
        self.gather = count
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None


class _DropWrite(_Gather):
    def __init__(self, dtype):
        self.outer = output = _ArrayState(dtype)
        self.inner = offsets = _ArrayState(dtype)
        self.prefix = prefix = _ArrayState(np.int64)

        def write(i):
            if prefix[0] >= 0:
                # Assignment targets the device array substituted by CCCL.
                # pylint: disable-next=unsupported-assignment-operation
                output[i] = np.int64(offsets[i]) - prefix[i]
            return np.int8(0)

        self.operation = None
        self.gather = write
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None
        self.discard = DiscardIterator()

    def __call__(self, output, offsets, prefix, length, stream):
        self.outer.bind(output)
        self.inner.bind(offsets)
        self.prefix.bind(prefix)
        try:
            self._launch(self.discard, length, stream)
        finally:
            self.outer.clear()
            self.inner.clear()
            self.prefix.clear()


class _Drop:
    def __init__(self, dtype, length_offsets, length_indexes):
        self.count = _DropCount(dtype, length_offsets, length_indexes)
        self.write = _DropWrite(dtype)
        self.scanner = None
        self.operation = None
        self.initial = np.asarray(0, dtype=np.int64)

    def __call__(self, output, indexes, offsets, length, stream):
        counts = cp.empty(length, dtype=cp.int64)
        prefix = cp.empty(length, dtype=cp.int64)
        self.count(counts, offsets, indexes, length, stream)
        if self.scanner is None:
            self.operation = make_op_adapter(OpKind.PLUS)
            self.scanner = make_inclusive_scan(
                d_in=counts, d_out=prefix, op=self.operation, init_value=self.initial
            )
        arguments = {
            "d_in": counts,
            "d_out": prefix,
            "num_items": length,
            "op": self.operation,
            "init_value": self.initial,
            "stream": stream,
        }
        storage_bytes = self.scanner(temp_storage=None, **arguments)
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        self.scanner(temp_storage=storage, **arguments)
        self.write(output, offsets, prefix, length, stream)
        # Reuse the first count as the error flag instead of allocating status.
        return counts[:1]


@lru_cache(maxsize=16)
def _drop_setup(device, context, dtype, length_offsets, length_indexes):
    return _Drop(dtype, length_offsets, length_indexes)


def drop_status(output, indexes, offsets, length_offsets, length_indexes, stream):
    with _lock:
        setup = _drop_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            offsets.dtype.str,
            length_offsets,
            length_indexes,
        )
        return setup(output, indexes, offsets, length_offsets, stream)
