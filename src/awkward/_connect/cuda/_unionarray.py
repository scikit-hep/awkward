# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Union kernels consuming the canonical device-resident int64** offset table."""

from functools import lru_cache
from threading import RLock

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    inclusive_scan,
    make_inclusive_scan,
    make_reduce_into,
    make_unary_transform,
    reduce_into,
    unary_transform,
)
from cuda.compute.op import make_op_adapter
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


class _Flatten:
    def __init__(self, tag_dtype, index_dtype, output_tag_dtype, length, combine):
        self.length, self.combine = length, combine
        self.tags = tags = _ArrayState(tag_dtype)
        self.index = index = _ArrayState(index_dtype)
        self.pointers = pointers = _ArrayState(np.int64)
        self.sizes = sizes = _ArrayState(np.int64)
        self.offsets = offsets = _ArrayState(np.int64)
        self.positions = positions = _ArrayState(np.int64)
        self.outtags = outtags = _ArrayState(output_tag_dtype)
        self.outindex = outindex = _ArrayState(np.int64)
        self.outoffsets = outoffsets = _ArrayState(np.int64)
        self.states = (
            tags,
            index,
            pointers,
            sizes,
            offsets,
            positions,
            outtags,
            outindex,
            outoffsets,
        )

        def size(row):
            pointer = pointers[tags[row]]
            at = np.int64(index[row])
            return _load_i64(pointer, at + 1) - _load_i64(pointer, at)

        def fill(row):
            # pylint: disable=unsupported-assignment-operation
            tag = tags[row]
            pointer = pointers[tag]
            at = np.int64(index[row])
            start, stop = _load_i64(pointer, at), _load_i64(pointer, at + 1)
            cursor = positions[row - 1] if row else np.int64(0)
            for item in range(start, stop):
                outtags[cursor], outindex[cursor] = tag, item
                cursor += 1
            outoffsets[row + 1] = offsets[row]
            return np.int8(0)

        self.size, self.fill = size, fill
        self.iterator = CountingIterator(np.int64(0))
        self.discard = DiscardIterator()
        self.transforms = {}
        self.algorithms = {}
        self.initial = np.asarray(0, dtype=np.int64)

    def transform(self, name, operation, output, stream):
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
            num_items=self.length,
            stream=stream,
        )

    def algorithm(self, name, factory, build, arguments):
        if name not in self.algorithms:
            prepared = factory(**build)
            storage_bytes = prepared(temp_storage=None, **arguments)
            self.algorithms[name] = (prepared, storage_bytes)
        prepared, storage_bytes = self.algorithms[name]
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        prepared(temp_storage=storage, **arguments)

    def run(self, tags, index, pointers, outputs):
        sizes = cp.empty(self.length, dtype=cp.int64)
        temporary = [sizes]
        bindings = [
            (self.tags, tags),
            (self.index, index),
            (self.pointers, pointers),
            (self.sizes, sizes),
        ]
        if self.combine:
            offsets = cp.empty(self.length, dtype=cp.int64)
            positions = cp.empty(self.length, dtype=cp.int64)
            zero = cp.zeros(1, dtype=cp.int64)
            temporary.extend((offsets, positions, zero))
            bindings.extend(
                zip(
                    (
                        self.offsets,
                        self.positions,
                        self.outtags,
                        self.outindex,
                        self.outoffsets,
                    ),
                    (offsets, positions, *outputs),
                    strict=True,
                )
            )
        for state, array in bindings:
            state.bind(array)
        stream = _Stream()
        try:
            self.transform("sizes", self.size, sizes, stream)
            if self.combine:
                positive = cp.maximum(sizes, 0)
                temporary.append(positive)
                for name, source, target in (
                    ("offsets", sizes, offsets),
                    ("positions", positive, positions),
                ):
                    build = {
                        "d_in": source,
                        "d_out": target,
                        "op": OpKind.PLUS,
                        "init_value": zero,
                    }
                    arguments = {**build, "num_items": self.length, "stream": stream}
                    self.algorithm(name, make_inclusive_scan, build, arguments)
                self.transform("fill", self.fill, self.discard, stream)
            else:
                build = {
                    "d_in": sizes,
                    "d_out": outputs[0][:1],
                    "op": OpKind.PLUS,
                    "h_init": self.initial,
                }
                arguments = {**build, "num_items": self.length, "stream": stream}
                self.algorithm("total", make_reduce_into, build, arguments)
        finally:
            for state in self.states:
                state.clear()


@lru_cache(maxsize=32)
def _flatten_setup(
    device, context, tag_dtype, index_dtype, output_tag_dtype, length, combine
):
    return _Flatten(tag_dtype, index_dtype, output_tag_dtype, length, combine)


_flatten_lock = RLock()


def _flatten(fromtags, fromindex, length, offsetsraws, outputs, combine):
    with _flatten_lock:
        setup = _flatten_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            fromtags.dtype.str,
            fromindex.dtype.str,
            outputs[0].dtype.str,
            length,
            combine,
        )
        setup.run(fromtags, fromindex, offsetsraws, outputs)


def flatten_length(total_length, fromtags, fromindex, length, offsetsraws):
    total_length[:1] = 0
    if length:
        _flatten(fromtags, fromindex, length, offsetsraws, (total_length,), False)


def flatten_combine(
    totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws
):
    tooffsets[:1] = 0
    if length:
        _flatten(
            fromtags, fromindex, length, offsetsraws, (totags, toindex, tooffsets), True
        )


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
