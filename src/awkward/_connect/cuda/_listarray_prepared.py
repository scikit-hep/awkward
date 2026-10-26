# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Prepared ListArray validation and element-parallel output operations."""

from functools import lru_cache
from threading import RLock

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    make_reduce_into,
    make_unary_transform,
)
from cuda.compute.op import make_op_adapter

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module
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


class _Prepared:
    def __init__(self, length, kind, initial):
        self.length = length
        self.chunks = (length + 63) // 64
        self.iterator = CountingIterator(np.int64(0))
        self.discard = DiscardIterator()
        self.transforms = {}
        self.reducer = None
        self.reduce_operation = make_op_adapter(kind)
        self.initial = np.asarray(initial, dtype=np.int64)

    def transform(self, name, operation, output, length, stream):
        if name not in self.transforms:
            adapter = make_op_adapter(operation)
            self.transforms[name] = (
                adapter,
                make_unary_transform(d_in=self.iterator, d_out=output, op=adapter),
            )
        adapter, prepared = self.transforms[name]
        prepared(
            d_in=self.iterator,
            d_out=output,
            op=adapter,
            num_items=length,
            stream=stream,
        )

    def find(self, result, stream):
        if self.chunks == 1:
            self.transform("check", self.check, result, 1, stream)
            return
        partials = cp.empty(self.chunks, dtype=cp.int64)
        self.transform("check", self.check, partials, self.chunks, stream)
        if self.reducer is None:
            self.reducer = make_reduce_into(
                d_in=partials,
                d_out=result,
                op=self.reduce_operation,
                h_init=self.initial,
            )
        arguments = {
            "d_in": partials,
            "d_out": result,
            "num_items": self.chunks,
            "op": self.reduce_operation,
            "h_init": self.initial,
            "stream": stream,
        }
        storage_bytes = self.reducer(temp_storage=None, **arguments)
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        self.reducer(temp_storage=storage, **arguments)

    def bind(self, arrays):
        for state, array in zip(self.states, arrays, strict=True):
            state.bind(array)

    def clear(self):
        for state in self.states:
            state.clear()


class _Advanced(_Prepared):
    def __init__(self, dtypes, length, lencontent):
        super().__init__(length, OpKind.MINIMUM, length)
        self.states = tuple(
            _ArrayState(dtype) for dtype in (*dtypes, np.int64, np.int64)
        )
        carry, advanced, starts, stops, indices, previous, first, status = self.states

        def check(chunk):
            for row in range(chunk * 64, min(chunk * 64 + 64, length)):
                start = np.int64(starts[row])
                stop = np.int64(stops[row])
                if stop < start or (start != stop and stop > lencontent):
                    return row
                index = indices[previous[row]]
                size = stop - start
                regular = index + size if index < 0 else index
                if regular < 0 or regular >= size:
                    return row
            return np.int64(length)

        def write(row):
            # CCCL replaces metadata proxies with device arrays.
            boundary = first[0]
            if row < boundary:
                start = np.int64(starts[row])
                size = np.int64(stops[row]) - start
                index = indices[previous[row]]
                if index < 0:
                    index += size
                carry[row] = start + index
                advanced[row] = row
            if row == 0:
                status[0] = boundary
                status[1] = np.int64(9223372036854775807)
                if boundary < length:
                    start = np.int64(starts[boundary])
                    stop = np.int64(stops[boundary])
                    if stop < start:
                        status[0] = -(boundary + 1)
                        status[1] = 1
                    elif start != stop and stop > lencontent:
                        status[0] = -(boundary + 1)
                        status[1] = 2
                    else:
                        status[1] = indices[previous[boundary]]
            return np.int8(0)

        self.check, self.write = check, write

    def __call__(self, arrays, stream):
        first, status = cp.empty(1, dtype=cp.int64), cp.empty(2, dtype=cp.int64)
        self.bind((*arrays, first, status))
        try:
            self.find(first, stream)
            self.transform("write", self.write, self.discard, self.length, stream)
        finally:
            self.clear()
        return status


class _Jagged(_Prepared):
    def __init__(self, dtypes, length, size):
        super().__init__(length, OpKind.MINIMUM, length)
        self.states = tuple(
            _ArrayState(dtype) for dtype in (*dtypes, np.int64, np.int64)
        )
        multistarts, multistops, single, carry, starts, stops, first, status = (
            self.states
        )

        def check(chunk):
            for row in range(chunk * 64, min(chunk * 64 + 64, length)):
                start = np.int64(starts[row])
                stop = np.int64(stops[row])
                if stop < start or stop - start != size:
                    return row
            return np.int64(length)

        def write(k):
            boundary = first[0]
            if size > 0:
                row = k // size
                j = k % size
                if row < boundary:
                    multistarts[k] = single[j]
                    multistops[k] = single[j + 1]
                    carry[k] = np.int64(starts[row]) + j
            if k == 0:
                status[0] = length
                status[1] = np.int64(9223372036854775807)
                if boundary < length:
                    status[0] = -(boundary + 1)
                    status[1] = 1 if stops[boundary] < starts[boundary] else 2
            return np.int8(0)

        self.check, self.write = check, write
        self.items = max(1, length * max(size, 0))

    def __call__(self, arrays, stream):
        first, status = cp.empty(1, dtype=cp.int64), cp.empty(2, dtype=cp.int64)
        self.bind((*arrays, first, status))
        try:
            self.find(first, stream)
            self.transform("write", self.write, self.discard, self.items, stream)
        finally:
            self.clear()
        return status


class _Spread(_Prepared):
    def __init__(self, dtypes, length):
        super().__init__(length, OpKind.MAXIMUM, 0)
        self.states = tuple(_ArrayState(dtype) for dtype in (*dtypes, np.int64))
        output, advanced, offsets, unordered = self.states

        def check(chunk):
            for row in range(chunk * 64, min(chunk * 64 + 64, length)):
                if offsets[row + 1] < offsets[row]:
                    return np.int64(1)
            return np.int64(0)

        def write(j):
            if unordered[0] == 0:
                # Upper bound selects the last repeated offset. Write only
                # inside the declared range, preserving origin and spare tail.
                if j >= offsets[0] and j < offsets[length]:
                    left, right = 0, length + 1
                    while left < right:
                        middle = (left + right) // 2
                        if offsets[middle] <= j:
                            left = middle + 1
                        else:
                            right = middle
                    output[j] = advanced[left - 1]
            else:
                # Preserve the sequential CPU's last writer for overlapping
                # intervals, including intervals following decreasing offsets.
                row = length - 1
                while row >= 0:
                    if offsets[row] <= j and j < offsets[row + 1]:
                        output[j] = advanced[row]
                        break
                    row -= 1
            return np.int8(0)

        self.check, self.write = check, write

    def __call__(self, arrays, stream):
        unordered = cp.empty(1, dtype=cp.int64)
        self.bind((*arrays, unordered))
        try:
            self.find(unordered, stream)
            self.transform("write", self.write, self.discard, arrays[0].size, stream)
        finally:
            self.clear()


@lru_cache(maxsize=32)
def _setup(device, context, kind, dtypes, length, parameter):
    if kind == "advanced":
        return _Advanced(dtypes, length, parameter)
    if kind == "jagged":
        return _Jagged(dtypes, length, parameter)
    return _Spread(dtypes, length)


def run(kind, arrays, length, parameter, stream):
    with _lock:
        setup = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            kind,
            tuple(array.dtype.str for array in arrays),
            length,
            parameter,
        )
        return setup(arrays, stream)
