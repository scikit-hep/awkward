# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CCCL ListOffsetArray kernels, preserving canonical errors and output tails."""

from dataclasses import dataclass

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    inclusive_scan,
    reduce_into,
    segmented_reduce,
    unary_transform,
)

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module
_SLICE_NONE = 9223372036854775807


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


def _prefix(counts):
    offsets = cp.zeros(counts.size + 1, dtype=cp.int64)
    if counts.size:
        inclusive_scan(
            d_in=counts,
            d_out=offsets[1:],
            op=OpKind.PLUS,
            init_value=np.asarray(0, dtype=np.int64),
            num_items=counts.size,
            stream=_Stream(),
        )
    return offsets


def flatten_offsets(tooffsets, outeroffsets, outeroffsetslen, inneroffsets):
    outer = outeroffsets[:outeroffsetslen]

    def gather(i):
        return inneroffsets[outer[i]]

    _transform(gather, outeroffsetslen, tooffsets)


def to_regular(size, fromoffsets, offsetslength):
    if offsetslength <= 1:
        size[:1] = 0
        return None
    offsets = fromoffsets[:offsetslength]
    n = offsetslength - 1
    bad = cp.empty(n, dtype=cp.int64)

    def check(i):
        count = np.int64(offsets[i + 1]) - np.int64(offsets[i])
        first = np.int64(offsets[1]) - np.int64(offsets[0])
        return i if count < 0 or count != first else np.int64(offsets.size - 1)

    _transform(check, n, bad)
    first_bad = cp.empty(1, dtype=cp.int64)
    reduce_into(
        d_in=bad,
        d_out=first_bad,
        op=OpKind.MINIMUM,
        h_init=np.asarray(n, dtype=np.int64),
        num_items=n,
        stream=_Stream(),
    )
    status = cp.empty(2, dtype=cp.int64)

    def finish(_):
        row = first_bad[0]
        first = np.int64(offsets[1]) - np.int64(offsets[0])
        size[0] = -1 if first < 0 else first
        status[0] = row
        status[1] = 0
        if row < offsets.size - 1:
            count = np.int64(offsets[row + 1]) - np.int64(offsets[row])
            status[1] = 1 if count < 0 else 2
        return np.int8(0)

    _transform(finish, 1)
    row, code = status.get().tolist()
    if code:
        message = (
            "offsets must be monotonically increasing"
            if code == 1
            else "cannot convert to RegularArray because subarray lengths are not regular"
        )
        return _Error(
            message.encode(),
            row,
            _SLICE_NONE,
            b"cuda.compute (awkward_ListOffsetArray_toRegularArray)",
        )
    return None


def rpad(toindex, fromoffsets, fromlength, target):
    if fromlength == 0:
        return
    offsets = fromoffsets[: fromlength + 1]
    parameters = cp.asarray([target], dtype=cp.int64)
    counts = cp.empty(fromlength, dtype=cp.int64)

    def count(i):
        size = np.int64(offsets[i + 1]) - np.int64(offsets[i])
        return max(size, parameters[0])

    _transform(count, fromlength, counts)
    positions = _prefix(counts)

    def write(i):
        start = np.int64(offsets[i])
        size = np.int64(offsets[i + 1]) - start
        j = 0
        while j < size:
            toindex[positions[i] + j] = start + j
            j += 1
        while j < parameters[0]:
            toindex[positions[i] + j] = -1
            j += 1
        return np.int8(0)

    _transform(write, fromlength)


def drop_none(tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes):
    if length_offsets == 0:
        return None
    offsets = fromoffsets[:length_offsets]
    parameters = cp.asarray([length_indexes], dtype=cp.int64)
    status = cp.empty(1, dtype=cp.int64)

    def check(_):
        return np.int64(offsets[-1] > parameters[0])

    _transform(check, 1, status)
    counts = cp.zeros(length_offsets, dtype=cp.int64)

    def count(i):
        total = np.int64(0)
        if status[0] == 0:
            j = np.int64(0) if i == 0 else np.int64(offsets[i - 1])
            while j < np.int64(offsets[i]):
                if noneindexes[j] < 0:
                    total += 1
                j += 1
        return total

    _transform(count, length_offsets, counts)
    positions = _prefix(counts)

    def write(i):
        if status[0] == 0:
            tooffsets[i] = np.int64(offsets[i]) - positions[i + 1]
        return np.int8(0)

    _transform(write, length_offsets)
    if status.get()[0]:
        return _Error(
            b"offsets[i] > len(content)",
            length_offsets - 1,
            _SLICE_NONE,
            b"cuda.compute (awkward_ListOffsetArray_drop_none_indexes)",
        )
    return None


def local_parents(nextparents, offsets, length, nextparents_length):
    if length == 0 or nextparents_length == 0:
        return
    offsets = offsets[: length + 1]
    parameters = cp.asarray([nextparents_length], dtype=cp.int64)

    def write(_):
        # Deprecated ABI compatibility path, with no live layout callers.
        # Preserve sequential overwrites even for overlapping offset intervals.
        initial = np.int64(offsets[0])
        i = 0
        while i < offsets.size - 1:
            j = np.int64(offsets[i]) - initial
            stop = np.int64(offsets[i + 1]) - initial
            while j < stop and j < parameters[0]:
                nextparents[j] = i
                j += 1
            i += 1
        return np.int8(0)

    _transform(write, 1)


def nonlocal_ranges(outstarts, outstops, distincts, lendistincts, outlength):
    if outlength == 0:
        return
    if lendistincts == 0:
        outstarts[:outlength] = 0
        outstops[:outlength] = 0
        return
    width = lendistincts // outlength
    starts = cp.arange(outlength, dtype=cp.int64) * width
    stops = starts + width
    source = distincts[:lendistincts]

    def position(i):
        return i + 1 if source[i] != -1 else np.int64(0)

    # Feed a plain array to segmented_reduce. Stateful transform iterators hit
    # inference and launch failures with the supported CCCL/Numba combination.
    positions = cp.empty(lendistincts, dtype=cp.int64)
    _transform(position, lendistincts, positions)
    segmented_reduce(
        d_in=positions,
        d_out=outstops[:outlength],
        num_segments=outlength,
        start_offsets_in=starts,
        end_offsets_in=stops,
        op=OpKind.MAXIMUM,
        h_init=np.asarray(0, dtype=np.int64),
        max_segment_size=width,
        stream=_Stream(),
    )
    outstarts[:outlength] = starts
    outstops[:outlength] = cp.maximum(outstops[:outlength], starts)
