# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Reusable CCCL offset transforms, without owning caller buffers."""

from functools import lru_cache
from threading import RLock

import numpy as np
from cuda.compute import CountingIterator, make_reduce_into, make_unary_transform
from cuda.compute.op import make_op_adapter

from awkward._nplikes.cupy import Cupy

cp = Cupy.instance()._module
# CCCL prepared transforms mutate their launch descriptors. Serialize host setup
# and launch, including factories that may share a globally cached descriptor.
_lock = RLock()


class _ArrayState:
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
        finally:
            # Kernel arguments have been copied at launch. Retain no array
            # owners or active pointers in the Python operator's captured state.
            self.outer.clear()
            self.inner.clear()


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
            starts[i] = start
            return stop

        self.operation = None
        self.gather = find_stop
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None


@lru_cache(maxsize=16)
def _ranges_setup(device, context, starts_dtype, source_dtype, stops_dtype, width):
    return _Ranges(starts_dtype, source_dtype, width)


def nonlocal_offsets(starts, stops, source, length, groups, stream):
    with _lock:
        setup = _ranges_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            starts.dtype.str,
            source.dtype.str,
            stops.dtype.str,
            length // groups,
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
    def __init__(self, offsets_dtype):
        self.outer = size = _ArrayState(np.int64)
        self.inner = offsets = _ArrayState(offsets_dtype)

        def check(i):
            first = np.int64(offsets[1]) - np.int64(offsets[0])
            count = np.int64(offsets[i + 1]) - np.int64(offsets[i])
            if i == 0:
                size[0] = -1 if first < 0 else first
            if count < 0:
                return -(i + 1)
            if count != first:
                return i + 1
            return np.int64(0)

        self.operation = None
        self.gather = check
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None
        self.reducer = None
        self.reduce_operation = None
        self.initial = np.asarray(0, dtype=np.int64)

    def validate(self, size, offsets, length, stream):
        errors = cp.empty(length - 1, dtype=cp.int64)
        status = cp.empty(1, dtype=cp.int64)
        self(errors, size, offsets, length - 1, stream)
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
            "num_items": length - 1,
            "op": self.reduce_operation,
            "h_init": self.initial,
            "stream": stream,
        }
        storage_bytes = self.reducer(temp_storage=None, **arguments)
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        self.reducer(temp_storage=storage, **arguments)
        return status


@lru_cache(maxsize=16)
def _regular_setup(device, context, offsets_dtype):
    return _Regular(offsets_dtype)


def regular_status(size, offsets, length, stream):
    with _lock:
        setup = _regular_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            offsets.dtype.str,
        )
        return setup.validate(size, offsets, length, stream)
