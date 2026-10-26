# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""ListArray kernels implemented with CCCL transforms, scans and reductions.

Error-capable calls return the same KernelError protocol as the CPU kernels.
One compact status read is required for their immediate Python result. Kernels
without error returns do not read device scalars to determine launch sizes.
"""

from dataclasses import dataclass

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    inclusive_scan,
    reduce_into,
    unary_transform,
)

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module
_SLICE_NONE = 9223372036854775807


@dataclass
class _Error:
    str: bytes
    id: int
    attempt: int
    filename: bytes


class _Stream:
    def __init__(self):
        self.stream = cp.cuda.get_current_stream()

    def __cuda_stream__(self):
        return (0, self.stream.ptr)


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


def _first_error(codes):
    result = cp.empty(1, dtype=cp.int64)

    def position(i):
        return i if codes[i] != 0 else np.int64(codes.size)

    positions = cp.empty(codes.size, dtype=cp.int64)
    _transform(position, codes.size, positions)
    reduce_into(
        d_in=positions,
        d_out=result,
        op=OpKind.MINIMUM,
        h_init=np.asarray(codes.size, dtype=np.int64),
        num_items=codes.size,
        stream=_Stream(),
    )
    return result


def _error(codes, attempts, first, messages, name):
    status = cp.empty(3, dtype=cp.int64)

    def pack(_):
        i = first[0]
        status[0] = i
        status[1] = codes[i] if i < codes.size else 0
        status[2] = attempts[i] if i < codes.size else _SLICE_NONE
        return np.int8(0)

    _transform(pack, 1)
    row, code, attempt = status.get().tolist()
    if code:
        return _Error(
            messages[code - 1].encode(), row, attempt, f"cuda.compute ({name})".encode()
        )
    return None


def _prepared_error(status, length, messages, name):
    tag, payload = status.get().tolist()
    if tag >= length:
        return None
    if tag < 0:
        row, code, attempt = -tag - 1, payload, _SLICE_NONE
    else:
        row, code, attempt = tag, 3, payload
    return _Error(
        messages[code - 1].encode(), row, attempt, f"cuda.compute ({name})".encode()
    )


def validity(starts, stops, length, lencontent):
    if length == 0:
        return None
    starts = starts[:length]
    stops = stops[:length]
    parameters = cp.asarray([lencontent], dtype=cp.int64)
    codes = cp.empty(length, dtype=cp.int64)
    attempts = cp.full(length, _SLICE_NONE, dtype=cp.int64)

    def check(i):
        start = np.int64(starts[i])
        stop = np.int64(stops[i])
        if start != stop:
            if start > stop:
                return np.int64(1)
            if start < 0:
                return np.int64(2)
            if stop > parameters[0]:
                return np.int64(3)
        return np.int64(0)

    _transform(check, length, codes)
    return _error(
        codes,
        attempts,
        _first_error(codes),
        ("start[i] > stop[i]", "start[i] < 0", "stop[i] > len(content)"),
        "awkward_ListArray_validity",
    )


def next_at(tocarry, fromstarts, fromstops, lenstarts, at):
    if lenstarts == 0:
        return None
    starts = fromstarts[:lenstarts]
    stops = fromstops[:lenstarts]
    parameters = cp.asarray([at], dtype=cp.int64)
    codes = cp.empty(lenstarts, dtype=cp.int64)
    attempts = cp.full(lenstarts, at, dtype=cp.int64)

    def check(i):
        size = np.int64(stops[i]) - np.int64(starts[i])
        index = parameters[0]
        if index < 0:
            index += size
        return np.int64(1) if index < 0 or index >= size else np.int64(0)

    _transform(check, lenstarts, codes)
    first = _first_error(codes)

    def write(i):
        if i < first[0]:
            size = np.int64(stops[i]) - np.int64(starts[i])
            index = parameters[0]
            if index < 0:
                index += size
            tocarry[i] = np.int64(starts[i]) + index
        return np.int8(0)

    _transform(write, lenstarts)
    return _error(
        codes,
        attempts,
        first,
        ("index out of range",),
        "awkward_ListArray_getitem_next_at",
    )


def next_array(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent,
):
    if lenstarts == 0:
        return None
    starts = fromstarts[:lenstarts]
    stops = fromstops[:lenstarts]
    indices = fromarray[:lenarray]
    parameters = cp.asarray([lencontent], dtype=cp.int64)
    codes = cp.empty(lenstarts, dtype=cp.int64)
    attempts = cp.full(lenstarts, _SLICE_NONE, dtype=cp.int64)
    written = cp.zeros(lenstarts, dtype=cp.int64)

    def check(i):
        start = np.int64(starts[i])
        stop = np.int64(stops[i])
        if stop < start:
            return np.int64(1)
        if start != stop and stop > parameters[0]:
            return np.int64(2)
        size = stop - start
        j = 0
        while j < indices.size:
            index = indices[j]
            if index < 0:
                index += size
            if index < 0 or index >= size:
                attempts[i] = indices[j]
                return np.int64(3)
            written[i] += 1
            j += 1
        return np.int64(0)

    _transform(check, lenstarts, codes)
    first = _first_error(codes)

    def write(i):
        if i <= first[0]:
            start = np.int64(starts[i])
            size = np.int64(stops[i]) - start
            j = 0
            while j < written[i]:
                index = indices[j]
                if index < 0:
                    index += size
                destination = i * indices.size + j
                tocarry[destination] = start + index
                toadvanced[destination] = j
                j += 1
        return np.int8(0)

    _transform(write, lenstarts)
    return _error(
        codes,
        attempts,
        first,
        ("stops[i] < starts[i]", "stops[i] > len(content)", "index out of range"),
        "awkward_ListArray_getitem_next_array",
    )


def next_array_advanced(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    fromadvanced,
    lenstarts,
    lencontent,
):
    if lenstarts == 0:
        return None
    if hasattr(fromstarts, "__cuda_array_interface__"):
        from awkward._connect.cuda._listarray_prepared import run

        status = run(
            "advanced",
            (tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced),
            lenstarts,
            lencontent,
            _Stream(),
        )
        return _prepared_error(
            status,
            lenstarts,
            ("stops[i] < starts[i]", "stops[i] > len(content)", "index out of range"),
            "awkward_ListArray_getitem_next_array_advanced",
        )
    starts = fromstarts[:lenstarts]
    stops = fromstops[:lenstarts]
    parameters = cp.asarray([lencontent], dtype=cp.int64)
    codes = cp.empty(lenstarts, dtype=cp.int64)
    attempts = cp.full(lenstarts, _SLICE_NONE, dtype=cp.int64)

    def check(i):
        start = np.int64(starts[i])
        stop = np.int64(stops[i])
        if stop < start:
            return np.int64(1)
        if start != stop and stop > parameters[0]:
            return np.int64(2)
        index = fromarray[fromadvanced[i]]
        size = stop - start
        regular = index + size if index < 0 else index
        if regular < 0 or regular >= size:
            attempts[i] = index
            return np.int64(3)
        return np.int64(0)

    _transform(check, lenstarts, codes)
    first = _first_error(codes)

    def write(i):
        if i < first[0]:
            start = np.int64(starts[i])
            size = np.int64(stops[i]) - start
            index = fromarray[fromadvanced[i]]
            if index < 0:
                index += size
            tocarry[i] = start + index
            toadvanced[i] = i
        return np.int8(0)

    _transform(write, lenstarts)
    return _error(
        codes,
        attempts,
        first,
        ("stops[i] < starts[i]", "stops[i] > len(content)", "index out of range"),
        "awkward_ListArray_getitem_next_array_advanced",
    )


def jagged_expand(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length,
):
    if length == 0:
        return None
    if hasattr(fromstarts, "__cuda_array_interface__"):
        from awkward._connect.cuda._listarray_prepared import run

        status = run(
            "jagged",
            (multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstops),
            length,
            jaggedsize,
            _Stream(),
        )
        return _prepared_error(
            status,
            length,
            ("stops[i] < starts[i]", "cannot fit jagged slice into nested list"),
            "awkward_ListArray_getitem_jagged_expand",
        )
    starts = fromstarts[:length]
    stops = fromstops[:length]
    parameters = cp.asarray([jaggedsize], dtype=cp.int64)
    codes = cp.empty(length, dtype=cp.int64)
    attempts = cp.full(length, _SLICE_NONE, dtype=cp.int64)

    def check(i):
        start = np.int64(starts[i])
        stop = np.int64(stops[i])
        if stop < start:
            return np.int64(1)
        if stop - start != parameters[0]:
            return np.int64(2)
        return np.int64(0)

    _transform(check, length, codes)
    first = _first_error(codes)

    def write(i):
        if i < first[0]:
            j = 0
            while j < parameters[0]:
                k = i * parameters[0] + j
                multistarts[k] = singleoffsets[j]
                multistops[k] = singleoffsets[j + 1]
                tocarry[k] = np.int64(starts[i]) + j
                j += 1
        return np.int8(0)

    _transform(write, length)
    return _error(
        codes,
        attempts,
        first,
        ("stops[i] < starts[i]", "cannot fit jagged slice into nested list"),
        "awkward_ListArray_getitem_jagged_expand",
    )


def jagged_apply(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen,
):
    if sliceouterlen == 0:
        tooffsets[0] = 0
        return None
    starts = fromstarts[:sliceouterlen]
    stops = fromstops[:sliceouterlen]
    ss = slicestarts[:sliceouterlen]
    se = slicestops[:sliceouterlen]
    parameters = cp.asarray([sliceinnerlen, contentlen], dtype=cp.int64)
    codes = cp.empty(sliceouterlen, dtype=cp.int64)
    attempts = cp.full(sliceouterlen, _SLICE_NONE, dtype=cp.int64)
    written = cp.zeros(sliceouterlen, dtype=cp.int64)

    def check(i):
        left = ss[i]
        right = se[i]
        if left == right:
            return np.int64(0)
        if right < left:
            return np.int64(1)
        if right > parameters[0]:
            attempts[i] = right
            return np.int64(2)
        start = np.int64(starts[i])
        stop = np.int64(stops[i])
        if stop < start:
            return np.int64(3)
        if start != stop and stop > parameters[1]:
            return np.int64(4)
        size = stop - start
        j = left
        while j < right:
            index = sliceindex[j]
            if index < -size or index >= size:
                attempts[i] = index
                return np.int64(5)
            written[i] += 1
            j += 1
        return np.int64(0)

    _transform(check, sliceouterlen, codes)
    positions = _prefix(written)
    first = _first_error(codes)

    def write(i):
        if i <= first[0]:
            tooffsets[i] = positions[i]
            start = np.int64(starts[i])
            size = np.int64(stops[i]) - start
            j = 0
            while j < written[i]:
                index = sliceindex[ss[i] + j]
                if index < 0:
                    index += size
                tocarry[positions[i] + j] = start + index
                j += 1
        if i == 0 and first[0] == codes.size:
            tooffsets[codes.size] = positions[codes.size]
        return np.int8(0)

    _transform(write, sliceouterlen)
    return _error(
        codes,
        attempts,
        first,
        (
            "jagged slice's stops[i] < starts[i]",
            "jagged slice's offsets extend beyond its content",
            "stops[i] < starts[i]",
            "stops[i] > len(content)",
            "index out of range",
        ),
        "awkward_ListArray_getitem_jagged_apply",
    )


def jagged_shrink(
    tocarry, tosmalloffsets, tolargeoffsets, slicestarts, slicestops, length, missing
):
    if length == 0:
        tosmalloffsets[0] = 0
        tolargeoffsets[0] = 0
        return
    starts = slicestarts[:length]
    stops = slicestops[:length]
    counts = cp.empty(length, dtype=cp.int64)

    def count(i):
        result = np.int64(0)
        j = starts[i]
        while j < stops[i]:
            if missing[j] >= 0:
                result += 1
            j += 1
        return result

    _transform(count, length, counts)
    positions = _prefix(counts)
    tosmalloffsets[: length + 1] = positions + starts[0]
    tolargeoffsets[: length + 1] = _prefix(stops - starts) + starts[0]

    def write(i):
        k = positions[i]
        j = starts[i]
        while j < stops[i]:
            if missing[j] >= 0:
                tocarry[k] = j
                k += 1
            j += 1
        return np.int8(0)

    _transform(write, length)


def rpad(toindex, fromstarts, fromstops, tostarts, tostops, target, length):
    if length == 0:
        return
    starts = fromstarts[:length]
    stops = fromstops[:length]
    parameters = cp.asarray([target], dtype=cp.int64)
    counts = cp.maximum(stops.astype(cp.int64) - starts.astype(cp.int64), target)
    positions = _prefix(counts)
    tostarts[:length] = positions[:-1]
    tostops[:length] = positions[1:]

    def write(i):
        size = np.int64(stops[i]) - np.int64(starts[i])
        start = np.int64(starts[i])
        j = 0
        while j < size:
            toindex[positions[i] + j] = start + j
            j += 1
        while j < parameters[0]:
            toindex[positions[i] + j] = -1
            j += 1
        return np.int8(0)

    _transform(write, length)


def spreadadvanced(toadvanced, fromadvanced, fromoffsets, lenstarts):
    if lenstarts == 0:
        return
    if hasattr(fromoffsets, "__cuda_array_interface__"):
        from awkward._connect.cuda._listarray_prepared import run

        run("spread", (toadvanced, fromadvanced, fromoffsets), lenstarts, 0, _Stream())
        return
    offsets = fromoffsets[: lenstarts + 1]

    def write(i):
        j = np.int64(offsets[i])
        while j < np.int64(offsets[i + 1]):
            toadvanced[j] = fromadvanced[i]
            j += 1
        return np.int8(0)

    _transform(write, lenstarts)


def next_range(tooffsets, tocarry, fromstarts, fromstops, lenstarts, start, stop, step):
    tooffsets[0] = 0
    if lenstarts == 0:
        return
    starts = fromstarts[:lenstarts]
    stops = fromstops[:lenstarts]
    parameters = cp.asarray([start, stop, step], dtype=cp.int64)
    regular_starts = cp.empty(lenstarts, dtype=cp.int64)
    counts = cp.empty(lenstarts, dtype=cp.int64)

    def count(i):
        size = np.int64(stops[i]) - np.int64(starts[i])
        left = parameters[0]
        right = parameters[1]
        stride = parameters[2]
        if stride > 0:
            if left == _SLICE_NONE:
                left = 0
            elif left < 0:
                left += size
            left = min(max(left, 0), size)
            if right == _SLICE_NONE:
                right = size
            elif right < 0:
                right += size
            right = min(max(right, 0), size)
            right = max(right, left)
            regular_starts[i] = left
            return np.int64(0) if right <= left else 1 + (right - left - 1) // stride
        if left == _SLICE_NONE:
            left = size - 1
        elif left < 0:
            left += size
        left = min(max(left, -1), size - 1)
        if right == _SLICE_NONE:
            right = -1
        elif right < 0:
            right += size
        right = min(max(right, -1), size - 1)
        right = min(right, left)
        regular_starts[i] = left
        if left <= right:
            return np.int64(0)
        if stride == -9223372036854775807 - 1:
            return np.int64(1)
        return 1 + (left - right - 1) // -stride

    _transform(count, lenstarts, counts)
    positions = _prefix(counts)
    tooffsets[: lenstarts + 1] = positions

    def write(i):
        j = 0
        start = np.int64(starts[i]) + regular_starts[i]
        while j < counts[i]:
            tocarry[positions[i] + j] = start + j * parameters[2]
            j += 1
        return np.int8(0)

    _transform(write, lenstarts)
