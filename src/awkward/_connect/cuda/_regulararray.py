# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""RegularArray CCCL kernels with canonical errors and device pointer tables."""

import math
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    make_reduce_into,
    make_unary_transform,
    unary_transform,
)
from cuda.compute.op import make_op_adapter
from numba import types
from numba.extending import intrinsic, register_jitable

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module
_SLICE_NONE = 9223372036854775807


@intrinsic
def _store_i64(typingctx, address, index, value):
    """Store through an entry of the canonical int64** carry-pointer table."""
    if not all(isinstance(t, types.Integer) for t in (address, index, value)):
        return None
    signature = types.void(address, index, value)

    def codegen(context, builder, sig, args):
        pointer = builder.inttoptr(
            context.cast(builder, args[0], sig.args[0], types.uintp),
            context.get_value_type(types.CPointer(types.int64)),
        )
        offset = context.cast(builder, args[1], sig.args[1], types.intp)
        item = context.cast(builder, args[2], sig.args[2], types.int64)
        builder.store(item, builder.gep(pointer, [offset]))
        return context.get_dummy_value()

    return signature, codegen


@register_jitable
def _choose(top, choose):
    if choose < 0 or top < choose:
        return np.int64(0)
    choose = min(choose, top - choose)
    result = np.int64(1)
    q = np.int64(1)
    while q <= choose:
        numerator = top - q + 1
        a, b = numerator, q
        while b:
            a, b = b, a % b
        # Cancel the denominator first, avoiding overflow in the product when
        # the resulting binomial coefficient fits int64.
        result = (result // (q // a)) * (numerator // a)
        q += 1
    return result


class _Stream:
    def __init__(self):
        self.stream = cp.cuda.get_current_stream()

    def __cuda_stream__(self):
        return (0, self.stream.ptr)


@dataclass
class _Error:
    str: bytes
    id: int
    attempt: int
    filename: bytes


def _transform(op, length, output=None):
    if length:
        unary_transform(
            d_in=CountingIterator(np.int64(0)),
            d_out=DiscardIterator() if output is None else output,
            op=op,
            num_items=length,
            stream=_Stream(),
        )


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


class _Regularize:
    def __init__(self, dtype, length, size):
        self.source = source = _ArrayState(dtype)
        self.output = output = _ArrayState(dtype)
        self.status = status = _ArrayState(np.int64)
        self.length = length

        def small(_):
            # pylint: disable=unsupported-assignment-operation
            status[0], status[1] = length, np.int64(_SLICE_NONE)
            for row in range(length):
                value = source[row]
                if value < 0:
                    value += size
                output[row] = value
                if value < 0 or value >= size:
                    status[0], status[1] = row, source[row]
                    break
            return np.int8(0)

        def combine(left, right):
            a = np.int64(length)
            if left < length:
                value = source[left]
                if value < 0:
                    value += size
                if value < 0 or value >= size:
                    a = left
            b = np.int64(length)
            if right < length:
                value = source[right]
                if value < 0:
                    value += size
                if value < 0 or value >= size:
                    b = right
            return min(a, b)

        def write(row):
            # The normalized failing entry is written; the suffix is untouched.
            # pylint: disable=unsupported-assignment-operation
            first = status[0]
            if row <= first:
                value = source[row]
                if value < 0:
                    value += size
                output[row] = value
            if row == 0:
                status[1] = source[first] if first < length else np.int64(_SLICE_NONE)
            return np.int8(0)

        self.small, self.combine, self.write = small, combine, write
        self.iterator = CountingIterator(np.int64(0))
        self.discard = DiscardIterator()
        self.transforms = {}
        self.reducer = None
        self.storage_bytes = None
        self.initial = np.asarray(length, dtype=np.int64)

    def transform(self, name, operation, length, stream):
        if name not in self.transforms:
            adapter = make_op_adapter(operation)
            self.transforms[name] = (
                adapter,
                make_unary_transform(
                    d_in=self.iterator, d_out=self.discard, op=adapter
                ),
            )
        adapter, prepared = self.transforms[name]
        prepared(
            d_in=self.iterator,
            d_out=self.discard,
            op=adapter,
            num_items=length,
            stream=stream,
        )

    def __call__(self, output, source):
        status = cp.empty(2, dtype=cp.int64)
        stream = _Stream()
        states = (self.output, self.source, self.status)
        for state, array in zip(states, (output, source, status), strict=True):
            state.bind(array)
        try:
            if self.length <= 64:
                self.transform("small", self.small, 1, stream)
            else:
                if self.reducer is None:
                    self.adapter = make_op_adapter(self.combine)
                    self.reducer = make_reduce_into(
                        d_in=self.iterator,
                        d_out=status[:1],
                        op=self.adapter,
                        h_init=self.initial,
                    )
                arguments = {
                    "d_in": self.iterator,
                    "d_out": status[:1],
                    "op": self.adapter,
                    "h_init": self.initial,
                    "num_items": self.length,
                    "stream": stream,
                }
                if self.storage_bytes is None:
                    self.storage_bytes = self.reducer(temp_storage=None, **arguments)
                storage = cp.empty(self.storage_bytes, dtype=cp.uint8)
                self.reducer(temp_storage=storage, **arguments)
                self.transform("write", self.write, self.length, stream)
        finally:
            for state in states:
                state.clear()
        return status


@lru_cache(maxsize=32)
def _regularize_setup(device, context, dtype, length, size):
    return _Regularize(dtype, length, size)


_regularize_lock = RLock()


def regularize(toarray, fromarray, lenarray, size):
    if lenarray == 0:
        return None
    with _regularize_lock:
        setup = _regularize_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            toarray.dtype.str,
            lenarray,
            size,
        )
        status = setup(toarray, fromarray)
    row, attempt = status.get().tolist()
    if row < lenarray:
        return _Error(
            b"index out of range",
            _SLICE_NONE,
            attempt,
            b"cuda.compute (awkward_RegularArray_getitem_next_array_regularize)",
        )
    return None


def local_parents(nextparents, size, length):
    if size <= 0 or length <= 0:
        return
    parameters = cp.asarray([size], dtype=cp.int64)

    def parent(i):
        return i // parameters[0]

    _transform(parent, size * length, nextparents)


def combinations(tocarry, toindex, fromindex, n, replacement, size, length):
    # These scalars are host metadata in the canonical kernel signature.
    n, size, length = int(n), int(size), int(length)
    effective = size + n - 1 if replacement else size
    per_row = math.comb(effective, n) if effective >= n and n >= 0 else 0
    total = length * per_row
    parameters = cp.asarray(
        [n, int(replacement), size, length, per_row, total], dtype=cp.int64
    )
    pointers = tocarry[:n]

    def fill(g):
        rank = g % parameters[4]
        row_start = (g // parameters[4]) * parameters[2]
        width = parameters[2] + parameters[0] - 1 if parameters[1] else parameters[2]
        lower = np.int64(0)
        position = np.int64(0)
        while position < parameters[0]:
            remaining = parameters[0] - position - 1
            combinations_left = _choose(width - lower, remaining + 1)
            lo = lower
            hi = width - remaining - 1
            # Cumulative subtree sizes locate the next candidate in logarithmic
            # time, using integers rather than a floating-point inverse.
            while lo < hi:
                mid = lo + (hi - lo + 1) // 2
                before = combinations_left - _choose(width - mid, remaining + 1)
                if before <= rank:
                    lo = mid
                else:
                    hi = mid - 1
            candidate = lo
            rank -= combinations_left - _choose(width - candidate, remaining + 1)
            value = candidate - position if parameters[1] else candidate
            _store_i64(pointers[position], g, row_start + value)
            lower = candidate + 1
            position += 1
        return np.int8(0)

    _transform(fill, total)

    def finish(k):
        toindex[k] = parameters[5]
        if parameters[3] > 0:
            if parameters[2] == 0:
                if k == 0:
                    fromindex[0] = 0
            else:
                stop = parameters[2] * parameters[3]
                fromindex[k] = stop if parameters[1] or k == 0 else stop + k - 1
        return np.int8(0)

    _transform(finish, n)
