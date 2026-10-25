# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Union kernels consuming the canonical device-resident int64** offset table."""

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    inclusive_scan,
    reduce_into,
    unary_transform,
)
from numba import types
from numba.extending import intrinsic

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module


@intrinsic
def _load_i64(typingctx, address, index):
    if not all(isinstance(t, types.Integer) for t in (address, index)):
        return None
    signature = types.int64(address, index)

    def codegen(context, builder, sig, args):
        pointer = builder.inttoptr(
            context.cast(builder, args[0], sig.args[0], types.uintp),
            context.get_value_type(types.CPointer(types.int64)),
        )
        offset = context.cast(builder, args[1], sig.args[1], types.intp)
        return builder.load(builder.gep(pointer, [offset]))

    return signature, codegen


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


def _scan(values, length, op=OpKind.PLUS):
    result = cp.empty(length, dtype=cp.int64)
    if length:
        inclusive_scan(
            d_in=values,
            d_out=result,
            op=op,
            init_value=cp.zeros(1, dtype=cp.int64),
            num_items=length,
            stream=_Stream(),
        )
    return result


def _sizes(tags, index, length, pointers):
    sizes = cp.empty(length, dtype=cp.int64)

    def size(i):
        pointer = pointers[tags[i]]
        position = np.int64(index[i])
        return _load_i64(pointer, position + 1) - _load_i64(pointer, position)

    _transform(size, length, sizes)
    return sizes


def flatten_length(total_length, fromtags, fromindex, length, offsetsraws):
    total_length[:1] = 0
    if length:
        sizes = _sizes(fromtags, fromindex, length, offsetsraws)
        reduce_into(
            d_in=sizes,
            d_out=total_length[:1],
            op=OpKind.PLUS,
            num_items=length,
            h_init=np.asarray(0, dtype=np.int64),
            stream=_Stream(),
        )


def flatten_combine(
    totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws
):
    tooffsets[:1] = 0
    if not length:
        return
    sizes = _sizes(fromtags, fromindex, length, offsetsraws)
    offsets = _scan(sizes, length)
    # The CPU advances its carry cursor only for nonempty ranges, even when
    # an offset pair decreases. Its signed output offsets still include that pair.
    positions = _scan(cp.maximum(sizes, 0), length)

    def fill(i):
        tag = fromtags[i]
        pointer = offsetsraws[tag]
        position = np.int64(fromindex[i])
        start = _load_i64(pointer, position)
        stop = _load_i64(pointer, position + 1)
        k = positions[i - 1] if i else np.int64(0)
        for j in range(start, stop):
            totags[k] = tag
            toindex[k] = j
            k += 1
        tooffsets[i + 1] = offsets[i]
        return np.int8(0)

    _transform(fill, length)


def nestedfill_tags_index(totags, toindex, tmpstarts, tag, fromcounts, length):
    if not length:
        return
    starts = tmpstarts[:length].copy()
    counts = fromcounts[:length]
    positions = _scan(cp.maximum(counts, 0), length)
    stops = starts + counts
    maxima = _scan(cp.where(counts > 0, stops, 0), length, OpKind.MAXIMUM)
    # Disjoint ordered ranges admit independent writes. Otherwise retain the
    # CPU's last-row-wins ordering without atomics or a host synchronization.
    overlap = cp.zeros(length, dtype=cp.int64)

    def check(i):
        return np.int64(i > 0 and counts[i] > 0 and starts[i] < maxima[i - 1])

    _transform(check, length, overlap)
    serial = cp.empty(1, dtype=cp.int64)
    reduce_into(
        d_in=overlap,
        d_out=serial,
        op=OpKind.MAXIMUM,
        num_items=length,
        h_init=np.asarray(0, dtype=np.int64),
        stream=_Stream(),
    )
    parameters = cp.asarray([tag, length], dtype=cp.int64)

    def fill(i):
        if serial[0] == 0:
            k = positions[i - 1] if i else np.int64(0)
            for j in range(starts[i], stops[i]):
                totags[j] = parameters[0]
                toindex[j] = k
                k += 1
        elif i == 0:
            k = np.int64(0)
            for row in range(parameters[1]):
                for j in range(starts[row], stops[row]):
                    totags[j] = parameters[0]
                    toindex[j] = k
                    k += 1
        tmpstarts[i] = stops[i]
        return np.int8(0)

    _transform(fill, length)
