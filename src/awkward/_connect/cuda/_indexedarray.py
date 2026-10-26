# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Prepared IndexedArray simplification with canonical first-error writes."""

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


class _Stream:
    def __init__(self):
        self.stream = cp.cuda.get_current_stream()

    def __cuda_stream__(self):
        return (0, self.stream.ptr)


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


class _Simplify:
    def __init__(self, output_dtype, outer_dtype, inner_dtype, length, innerlength):
        self.output = output = _ArrayState(output_dtype)
        self.outer = outer = _ArrayState(outer_dtype)
        self.inner = inner = _ArrayState(inner_dtype)
        self.status = status = _ArrayState(np.int64)
        chunk_size = 64

        def small(_):
            # CCCL substitutes device arrays for metadata proxies.
            # pylint: disable=unsupported-assignment-operation
            status[0] = length
            status[1] = np.int64(9223372036854775807)
            for row in range(length):
                index = np.int64(outer[row])
                if index >= 0 and index >= innerlength:
                    status[0] = row
                    status[1] = index
                    break
                output[row] = -1 if index < 0 else np.int64(inner[index])
            return np.int8(0)

        def check(chunk):
            start = chunk * chunk_size
            stop = min(start + chunk_size, length)
            for row in range(start, stop):
                index = np.int64(outer[row])
                if index >= 0 and index >= innerlength:
                    return row
            return np.int64(length)

        def write(row):
            # pylint: disable=unsupported-assignment-operation
            first = status[0]
            if row < first:
                index = np.int64(outer[row])
                output[row] = -1 if index < 0 else np.int64(inner[index])
            if row == 0:
                status[1] = (
                    np.int64(outer[first])
                    if first < length
                    else np.int64(9223372036854775807)
                )
            return np.int8(0)

        self.small, self.check, self.write = small, check, write
        self.length = length
        self.chunks = (length + chunk_size - 1) // chunk_size
        self.iterator = CountingIterator(np.int64(0))
        self.discard = DiscardIterator()
        self.transforms = {}
        self.reducer = None
        self.reduce_operation = None
        self.initial = np.asarray(length, dtype=np.int64)

    def transform(self, name, operation, output, length, stream):
        if name not in self.transforms:
            adapter = make_op_adapter(operation)
            prepared = make_unary_transform(
                d_in=self.iterator, d_out=output, op=adapter
            )
            self.transforms[name] = (adapter, prepared)
        adapter, prepared = self.transforms[name]
        prepared(
            d_in=self.iterator,
            d_out=output,
            op=adapter,
            num_items=length,
            stream=stream,
        )

    def __call__(self, output, outer, inner, stream):
        status = cp.empty(2, dtype=cp.int64)
        states = (self.output, self.outer, self.inner, self.status)
        for state, array in zip(states, (output, outer, inner, status), strict=True):
            state.bind(array)
        try:
            if self.length <= 64:
                self.transform("small", self.small, self.discard, 1, stream)
            else:
                positions = cp.empty(self.chunks, dtype=cp.int64)
                self.transform("check", self.check, positions, self.chunks, stream)
                first = status[:1]
                if self.reducer is None:
                    self.reduce_operation = make_op_adapter(OpKind.MINIMUM)
                    self.reducer = make_reduce_into(
                        d_in=positions,
                        d_out=first,
                        op=self.reduce_operation,
                        h_init=self.initial,
                    )
                arguments = {
                    "d_in": positions,
                    "d_out": first,
                    "num_items": self.chunks,
                    "op": self.reduce_operation,
                    "h_init": self.initial,
                    "stream": stream,
                }
                storage_bytes = self.reducer(temp_storage=None, **arguments)
                storage = cp.empty(storage_bytes, dtype=cp.uint8)
                self.reducer(temp_storage=storage, **arguments)
                self.transform("write", self.write, self.discard, self.length, stream)
        finally:
            for state in states:
                state.clear()
        return status


@lru_cache(maxsize=32)
def _setup(
    device, context, output_dtype, outer_dtype, inner_dtype, length, innerlength
):
    return _Simplify(output_dtype, outer_dtype, inner_dtype, length, innerlength)


def simplify_status(output, outer, length, inner, innerlength):
    with _lock:
        setup = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            output.dtype.str,
            outer.dtype.str,
            inner.dtype.str,
            length,
            innerlength,
        )
        return setup(output, outer, inner, _Stream())
