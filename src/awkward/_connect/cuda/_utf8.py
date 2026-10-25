# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""UTF-8 to padded UTF-32 conversion with the CPU kernel's byte semantics."""

from dataclasses import dataclass

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    reduce_into,
    unary_transform,
)
from numba.extending import register_jitable

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module


@register_jitable
def _width(byte):
    if byte & 0x80 == 0:
        return 1
    if byte & 0xE0 == 0xC0:
        return 2
    if byte & 0xF0 == 0xE0:
        return 3
    if byte & 0xF8 == 0xF0:
        return 4
    return 0


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


def prepare(fromptr, fromoffsets, offsetslength, outmaxcodepoints):
    rows = max(0, offsetslength - 1)
    outmaxcodepoints[:1] = 0
    if not rows:
        return
    counts = cp.empty(rows, dtype=cp.int64)

    def count(row):
        position = np.int64(fromoffsets[row])
        stop = np.int64(fromoffsets[row + 1])
        result = np.int64(0)
        while position < stop:
            width = _width(fromptr[position])
            # The canonical preparation contract requires valid leading bytes;
            # the CPU makes no progress on a zero-width byte. Avoid a GPU hang.
            if width == 0:
                break
            position += width
            result += 1
        return result

    _transform(count, rows, counts)
    reduce_into(
        d_in=counts,
        d_out=outmaxcodepoints[:1],
        op=OpKind.MAXIMUM,
        num_items=rows,
        h_init=np.asarray(0, dtype=np.int64),
        stream=_Stream(),
    )


def decode(fromptr, fromoffsets, offsetslength, maxcodepoints, toptr):
    rows = max(0, offsetslength - 1)
    if not rows:
        return None
    bad = cp.empty(rows, dtype=cp.int64)
    parameters = cp.asarray([maxcodepoints, rows], dtype=cp.int64)

    def check(row):
        position = np.int64(fromoffsets[row])
        stop = np.int64(fromoffsets[row + 1])
        while position < stop:
            width = _width(fromptr[position])
            if width == 0:
                return row
            position += width
        return parameters[1]

    _transform(check, rows, bad)
    first = cp.empty(1, dtype=cp.int64)
    reduce_into(
        d_in=bad,
        d_out=first,
        op=OpKind.MINIMUM,
        num_items=rows,
        h_init=np.asarray(rows, dtype=np.int64),
        stream=_Stream(),
    )
    status = cp.zeros(2, dtype=cp.int64)

    def fill(row):
        if row <= first[0]:
            position = np.int64(fromoffsets[row])
            stop = np.int64(fromoffsets[row + 1])
            output = row * parameters[0]
            end = output + parameters[0]
            while position < stop:
                byte = np.int64(fromptr[position])
                width = _width(byte)
                if width == 0:
                    status[0] = 1
                    status[1] = byte
                    return np.int8(0)
                value = byte & (0x7F if width == 1 else (1 << (7 - width)) - 1)
                for j in range(1, width):
                    value = (value << 6) | (np.int64(fromptr[position + j]) & 0x3F)
                toptr[output] = value
                output += 1
                position += width
            while output < end:
                toptr[output] = 0
                output += 1
        return np.int8(0)

    _transform(fill, rows)
    failed, byte = status.get().tolist()
    if failed:
        return _Error(
            b"could not convert UTF8 code point to UTF32: invalid byte in UTF8 string",
            9223372036854775807,
            byte,
            b"cuda.compute (awkward_NumpyArray_utf8_to_utf32_padded)",
        )
    return None
