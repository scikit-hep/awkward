# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""RegularArray CCCL kernels with canonical errors and device pointer tables."""

import math
from dataclasses import dataclass

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    reduce_into,
    unary_transform,
)
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


def regularize(toarray, fromarray, lenarray, size):
    if lenarray == 0:
        return None
    source = fromarray[:lenarray]
    parameters = cp.asarray([size], dtype=cp.int64)
    values = cp.empty(lenarray, dtype=cp.int64)
    bad = cp.empty(lenarray, dtype=cp.int64)

    def check(i):
        value = source[i]
        if value < 0:
            value += parameters[0]
        values[i] = value
        return i if value < 0 or value >= parameters[0] else np.int64(source.size)

    _transform(check, lenarray, bad)
    first = cp.empty(1, dtype=cp.int64)
    reduce_into(
        d_in=bad,
        d_out=first,
        op=OpKind.MINIMUM,
        num_items=lenarray,
        h_init=np.asarray(lenarray, dtype=np.int64),
        stream=_Stream(),
    )
    status = cp.empty(2, dtype=cp.int64)

    def write(i):
        # The CPU writes the normalized failing entry before returning an error.
        if i <= first[0]:
            toarray[i] = values[i]
        if i == 0:
            row = first[0]
            status[0] = row
            status[1] = source[row] if row < source.size else _SLICE_NONE
        return np.int8(0)

    _transform(write, lenarray)
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
