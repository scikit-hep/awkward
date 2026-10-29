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

    @property
    def size(self):
        return len(self)

    def clear(self):
        self.__cuda_array_interface__ = {
            "shape": (0,),
            "strides": None,
            "typestr": self.dtype.str,
            "data": (0, False),
            "version": 3,
        }


class _Prepared:
    def __init__(self, length, kind, initial, chunk_size=64):
        self.length = length
        self.chunk_size = chunk_size
        self.chunks = (length + chunk_size - 1) // chunk_size
        self.iterator = CountingIterator(np.int64(0))
        self.discard = DiscardIterator()
        self.transforms = {}
        self.reducer = None
        self.reduce_storage_bytes = None
        self.reduce_operation = make_op_adapter(kind)
        self.initial = np.asarray(initial, dtype=np.int64)
        self.runtime = _ArrayState(np.int64)
        self.runtime_array = None
        self.workspace_sizes = {}

    def configure(self, length, parameter, capacity=0):
        self.length = length
        self.chunks = (length + self.chunk_size - 1) // self.chunk_size
        self.runtime_values = (length, parameter)
        if isinstance(self, _Spread):
            self.runtime_values = (length, capacity)
        else:
            self.initial = np.asarray(length, dtype=np.int64)
        if isinstance(self, _Jagged):
            self.items = max(1, length * max(parameter, 0))

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

    def find(self, result, stream, partials=None):
        if self.chunks == 1:
            self.transform("check", self.check, result, 1, stream)
            return
        if partials is None:
            partials = cp.empty(self.chunks, dtype=cp.int64)
        self.transform("check", self.check, partials, self.chunks, stream)
        self.reduce(partials, result, self.chunks, stream)

    def reduce(self, inputs, result, length, stream):
        if self.reducer is None:
            self.reducer = make_reduce_into(
                d_in=inputs,
                d_out=result,
                op=self.reduce_operation,
                h_init=self.initial,
            )
        arguments = {
            "d_in": inputs,
            "d_out": result,
            "num_items": length,
            "op": self.reduce_operation,
            "h_init": self.initial,
            "stream": stream,
        }
        if length not in self.workspace_sizes:
            if len(self.workspace_sizes) >= 32:
                self.workspace_sizes.pop(next(iter(self.workspace_sizes)))
            self.workspace_sizes[length] = self.reducer(temp_storage=None, **arguments)
        self.reduce_storage_bytes = self.workspace_sizes[length]
        storage = cp.empty(self.reduce_storage_bytes, dtype=cp.uint8)
        self.reducer(temp_storage=storage, **arguments)

    def bind(self, arrays):
        # Values vary per invocation; only dtype/rank enter CCCL's JIT key.
        # Retain the upload until all dependent work has been enqueued.
        self.runtime_array = cp.asarray(self.runtime_values, dtype=cp.int64)
        self.runtime.bind(self.runtime_array)
        for state, array in zip(self.states, arrays, strict=True):
            state.bind(array)

    def clear(self):
        for state in self.states:
            state.clear()
        self.runtime.clear()
        self.runtime_array = None


class _FirstError(_Prepared):
    def __init__(self, dtype):
        super().__init__(0, OpKind.MINIMUM, 0)
        codes = _ArrayState(dtype)
        self.states = (codes,)

        def earliest(left, right):
            a, b = np.int64(codes.size), np.int64(codes.size)
            if left < codes.size and codes[left] != 0:
                a = left
            if right < codes.size and codes[right] != 0:
                b = right
            return min(a, b)

        # The map is idempotent, so min(map(a), map(b)) is associative on
        # rows plus the sentinel. No stateful TransformIterator is constructed.
        self.reduce_operation = make_op_adapter(earliest)

    def __call__(self, arrays, stream):
        result = cp.empty(1, dtype=cp.int64)
        self.states[0].bind(arrays[0])
        try:
            self.reduce(self.iterator, result, arrays[0].size, stream)
        finally:
            self.clear()
        return result


class _NextArray(_Prepared):
    def __init__(self, dtypes):
        super().__init__(0, OpKind.MINIMUM, 0)
        self.states = tuple(
            _ArrayState(dtype) for dtype in (*dtypes, np.int64, np.int64)
        )
        carry, advanced, starts, stops, indices, first, status = self.states
        runtime = self.runtime

        def check(k):
            chunks = max(1, (indices.size + 63) // 64)
            row, chunk = k // chunks, k % chunks
            start, stop = np.int64(starts[row]), np.int64(stops[row])
            stride = indices.size + 1
            if stop < start or (start != stop and stop > runtime[1]):
                return row * stride
            size = stop - start
            for j in range(chunk * 64, min((chunk + 1) * 64, indices.size)):
                index = indices[j]
                if index < -size or index >= size:
                    return row * stride + j
            return runtime[0] * stride

        def write(k):
            stride = indices.size + 1
            if indices.size:
                row, j = k // indices.size, k % indices.size
                if row * stride + j < first[0]:
                    start = np.int64(starts[row])
                    size = np.int64(stops[row]) - start
                    index = indices[j]
                    if index < 0:
                        index += size
                    carry[k] = start + index
                    advanced[k] = j
            if k == 0:
                row, j = first[0] // stride, first[0] % stride
                status[0], status[1] = row, 0
                status[2] = np.int64(9223372036854775807)
                if row < runtime[0]:
                    start, stop = np.int64(starts[row]), np.int64(stops[row])
                    if stop < start:
                        status[1] = 1
                    elif start != stop and stop > runtime[1]:
                        status[1] = 2
                    else:
                        status[1] = 3
                        status[2] = indices[j]
            return np.int8(0)

        self.check, self.write = check, write

    def __call__(self, arrays, stream):
        columns = arrays[4].size
        self.chunks = self.length * max(1, (columns + 63) // 64)
        self.initial = np.asarray(self.length * (columns + 1), dtype=np.int64)
        arena = cp.empty(4 + self.chunks, dtype=cp.int64)
        status, first = arena[:3], arena[3:4]
        self.bind((*arrays, first, status))
        try:
            self.find(first, stream, arena[4:])
            self.transform(
                "write", self.write, self.discard, max(1, self.length * columns), stream
            )
        finally:
            self.clear()
        return status


class _Advanced(_Prepared):
    def __init__(self, dtypes, length, lencontent, chunk_size=64):
        super().__init__(length, OpKind.MINIMUM, length, chunk_size)
        self.states = tuple(
            _ArrayState(dtype) for dtype in (*dtypes, np.int64, np.int64)
        )
        carry, advanced, starts, stops, indices, previous, first, status = self.states

        runtime = self.runtime
        self.runtime_values = (
            length,
            lencontent,
        )

        def check(chunk):
            for row in range(
                chunk * chunk_size, min((chunk + 1) * chunk_size, runtime[0])
            ):
                start = np.int64(starts[row])
                stop = np.int64(stops[row])
                if stop < start or (start != stop and stop > runtime[1]):
                    return row
                index = indices[previous[row]]
                size = stop - start
                regular = index + size if index < 0 else index
                if regular < 0 or regular >= size:
                    return row
            return np.int64(runtime[0])

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
                if boundary < runtime[0]:
                    start = np.int64(starts[boundary])
                    stop = np.int64(stops[boundary])
                    if stop < start:
                        status[0] = -(boundary + 1)
                        status[1] = 1
                    elif start != stop and stop > runtime[1]:
                        status[0] = -(boundary + 1)
                        status[1] = 2
                    else:
                        status[1] = indices[previous[boundary]]
            return np.int8(0)

        def small(_):
            status[0] = runtime[0]
            status[1] = np.int64(9223372036854775807)
            for row in range(runtime[0]):
                start = np.int64(starts[row])
                stop = np.int64(stops[row])
                if stop < start:
                    status[0] = -(row + 1)
                    status[1] = 1
                    break
                if start != stop and stop > runtime[1]:
                    status[0] = -(row + 1)
                    status[1] = 2
                    break
                index = indices[previous[row]]
                size = stop - start
                regular = index + size if index < 0 else index
                if regular < 0 or regular >= size:
                    status[0] = row
                    status[1] = index
                    break
                carry[row] = start + regular
                advanced[row] = row
            return np.int8(0)

        self.small = small
        self.check, self.write = check, write

    def __call__(self, arrays, stream):
        status = cp.empty(2, dtype=cp.int64)
        first = status[:1] if self.length <= 64 else cp.empty(1, dtype=cp.int64)
        self.bind((*arrays, first, status))
        try:
            if self.length <= 64:
                self.transform("small", self.small, self.discard, 1, stream)
            else:
                self.find(first, stream)
                self.transform("write", self.write, self.discard, self.length, stream)
        finally:
            self.clear()
        return status


class _Jagged(_Prepared):
    def __init__(self, dtypes, length, size, chunk_size=64):
        super().__init__(length, OpKind.MINIMUM, length, chunk_size)
        self.states = tuple(_ArrayState(dtype) for dtype in (*dtypes, np.int64))
        multistarts, multistops, single, carry, starts, stops, status = self.states

        runtime = self.runtime
        self.runtime_values = (
            length,
            size,
        )

        def check(chunk):
            for row in range(
                chunk * chunk_size, min((chunk + 1) * chunk_size, runtime[0])
            ):
                start = np.int64(starts[row])
                stop = np.int64(stops[row])
                if stop < start or stop - start != runtime[1]:
                    return row
            return np.int64(runtime[0])

        def write(k):
            boundary = status[2]
            if runtime[1] > 0:
                row = k // runtime[1]
                j = k % runtime[1]
                if row < boundary:
                    multistarts[k] = single[j]
                    multistops[k] = single[j + 1]
                    carry[k] = np.int64(starts[row]) + j
            if k == 0:
                status[0] = runtime[0]
                status[1] = np.int64(9223372036854775807)
                if boundary < runtime[0]:
                    status[0] = -(boundary + 1)
                    status[1] = 1 if stops[boundary] < starts[boundary] else 2
            return np.int8(0)

        def small(k):
            boundary = runtime[0]
            for row in range(runtime[0]):
                start = np.int64(starts[row])
                stop = np.int64(stops[row])
                if stop < start or stop - start != runtime[1]:
                    boundary = row
                    break
            if runtime[1] > 0:
                row, j = k // runtime[1], k % runtime[1]
                if row < boundary:
                    multistarts[k] = single[j]
                    multistops[k] = single[j + 1]
                    carry[k] = np.int64(starts[row]) + j
            if k == 0:
                status[0] = runtime[0]
                status[1] = np.int64(9223372036854775807)
                if boundary < runtime[0]:
                    status[0] = -(boundary + 1)
                    status[1] = 1 if stops[boundary] < starts[boundary] else 2
            return np.int8(0)

        self.small = small
        self.check, self.write = check, write
        self.items = max(1, length * max(size, 0))

    def __call__(self, arrays, stream):
        # Bound both repeated validation and total output work. Long outputs
        # retain the measured element-parallel validation/write pipeline.
        small = self.length <= 64 and self.items <= 256
        # One allocation for status, the first-error row and chunk results.
        # Each invocation owns its arena; cached operators never retain it.
        arena = cp.empty(2 if small else 3 + self.chunks, dtype=cp.int64)
        status = arena[:2]
        first = status[:1] if small else arena[2:3]
        self.bind((*arrays, arena))
        try:
            if small:
                self.transform("small", self.small, self.discard, self.items, stream)
            else:
                self.find(first, stream, arena[3:])
                self.transform("write", self.write, self.discard, self.items, stream)
        finally:
            self.clear()
        return status


class _Spread(_Prepared):
    def __init__(self, dtypes, length, chunk_size=64, lanes=0, capacity=0):
        super().__init__(length, OpKind.MAXIMUM, 0, chunk_size)
        self.states = tuple(_ArrayState(dtype) for dtype in (*dtypes, np.int64))
        output, advanced, offsets, unordered = self.states

        runtime = self.runtime
        self.runtime_values = (
            length,
            capacity,
        )

        def check(chunk):
            for row in range(
                chunk * chunk_size, min((chunk + 1) * chunk_size, runtime[0])
            ):
                if offsets[row + 1] < offsets[row]:
                    return np.int64(1)
            return np.int64(0)

        def write(j):
            if unordered[0] == 0:
                # Upper bound selects the last repeated offset. Write only
                # inside the declared range, preserving origin and spare tail.
                if j >= offsets[0] and j < offsets[runtime[0]]:
                    left, right = 0, runtime[0] + 1
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
                row = runtime[0] - 1
                while row >= 0:
                    if offsets[row] <= j and j < offsets[row + 1]:
                        output[j] = advanced[row]
                        break
                    row -= 1
            return np.int8(0)

        def small(j):
            # Bounded reverse lookup fuses ownership resolution with writing.
            # It is deterministic for overlapping intervals as well as repeats.
            row = runtime[0] - 1
            while row >= 0:
                if offsets[row] <= j and j < offsets[row + 1]:
                    output[j] = advanced[row]
                    break
                row -= 1
            return np.int8(0)

        def tiled(k):
            if unordered[0] == 0:
                row, lane = k // lanes, k % lanes
                if row < runtime[0]:
                    start = np.int64(offsets[row])
                    stop = np.int64(offsets[row + 1])
                    if stop > start and lane < stop - start:
                        j = start + lane
                        while j < stop:
                            output[j] = advanced[row]
                            if stop - j <= lanes:
                                break
                            j += lanes
            elif k < runtime[1]:
                row = runtime[0] - 1
                while row >= 0:
                    if offsets[row] <= k and k < offsets[row + 1]:
                        output[k] = advanced[row]
                        break
                    row -= 1
            return np.int8(0)

        self.tiled = tiled
        self.lanes = lanes
        self.small = small
        self.check, self.write = check, write

    def __call__(self, arrays, stream):
        # Keep the measured parallel binary-search path for large outputs.
        small = self.lanes == 0 and self.length <= 64 and arrays[0].size <= 1024
        unordered = arrays[0] if small else cp.empty(1, dtype=cp.int64)
        self.bind((*arrays, unordered))
        try:
            if small:
                self.transform(
                    "small", self.small, self.discard, arrays[0].size, stream
                )
            else:
                self.find(unordered, stream)
                if self.lanes:
                    self.transform(
                        "tiled",
                        self.tiled,
                        self.discard,
                        max(self.length * self.lanes, arrays[0].size),
                        stream,
                    )
                else:
                    self.transform(
                        "write", self.write, self.discard, arrays[0].size, stream
                    )
        finally:
            self.clear()


@lru_cache(maxsize=32)
def _setup(
    device,
    context,
    kind,
    dtypes,
    chunk_size=64,
    spread_lanes=0,
):
    if kind == "first_error":
        return _FirstError(dtypes[0])
    if kind == "array":
        return _NextArray(dtypes)
    if kind == "advanced":
        return _Advanced(dtypes, 0, 0, chunk_size)
    if kind == "jagged":
        return _Jagged(dtypes, 0, 0, chunk_size)
    return _Spread(dtypes, 0, chunk_size, spread_lanes, 0)


def run(kind, arrays, length, parameter, stream, *, chunk_size=64, spread_lanes=0):
    if chunk_size < 1 or spread_lanes < 0:
        raise ValueError("chunk size must be positive and lanes nonnegative")
    with _lock:
        setup = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            kind,
            tuple(array.dtype.str for array in arrays),
            chunk_size,
            spread_lanes,
        )
        setup.configure(length, parameter, arrays[0].size)
        return setup(arrays, stream)
