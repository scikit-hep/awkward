# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Union kernels consuming the canonical device-resident int64** offset table."""

import ctypes
from functools import lru_cache
from threading import RLock

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    make_inclusive_scan,
    make_reduce_into,
    make_unary_transform,
    unary_transform,
)
from cuda.compute.op import make_op_adapter
from numba_cuda_mlir import cuda as numba_cuda

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module


@numba_cuda.jit(device=True)
def _load_i64(address, index):
    pointer = ctypes.cast(address, ctypes.POINTER(ctypes.c_int64))
    return pointer[index]


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


class _NestedFill:
    transform = _Flatten.transform
    algorithm = _Flatten.algorithm

    def __init__(self, tag_dtype, index_dtype, starts_dtype, counts_dtype, tag, length):
        self.length = length
        self.tags = tags = _ArrayState(tag_dtype)
        self.index = index = _ArrayState(index_dtype)
        self.scratch = scratch = _ArrayState(starts_dtype)
        self.counts = counts = _ArrayState(counts_dtype)
        self.starts = starts = _ArrayState(starts_dtype)
        self.stops = stops = _ArrayState(starts_dtype)
        self.positive = positive = _ArrayState(np.int64)
        self.present_stops = present_stops = _ArrayState(np.int64)
        self.positions = positions = _ArrayState(np.int64)
        self.maxima = maxima = _ArrayState(np.int64)
        self.serial = serial = _ArrayState(np.int64)
        self.states = (
            tags,
            index,
            scratch,
            counts,
            starts,
            stops,
            positive,
            present_stops,
            positions,
            maxima,
            serial,
        )

        def prepare(row):
            # Fuse the old copy/add/maximum/where launches into one pass.
            # pylint: disable=unsupported-assignment-operation
            start, count = scratch[row], counts[row]
            starts[row], stops[row] = start, start + count
            positive[row] = max(count, 0)
            present_stops[row] = start + count if count > 0 else 0
            return np.int8(0)

        def prepare_small(_):
            # Bounded row metadata work only; output writes remain parallel.
            # pylint: disable=unsupported-assignment-operation
            rank, maximum, overlaps = np.int64(0), np.int64(0), np.int64(0)
            for row in range(length):
                start, count = scratch[row], counts[row]
                stop = start + count
                starts[row], stops[row] = start, stop
                if row > 0 and count > 0 and start < maximum:
                    overlaps = np.int64(1)
                maximum = max(maximum, stop if count > 0 else 0)
                rank += max(count, 0)
                positions[row] = rank
            serial[0] = overlaps
            return np.int8(0)

        def check(row):
            return np.int64(
                row > 0 and counts[row] > 0 and starts[row] < maxima[row - 1]
            )

        def fill(row):
            # Disjoint rows write in parallel. Overlapping/unordered ranges use
            # the canonical row order, retaining last-row-wins behavior.
            # pylint: disable=unsupported-assignment-operation
            if serial[0] == 0:
                rank = positions[row - 1] if row else np.int64(0)
                for item in range(starts[row], stops[row]):
                    tags[item], index[item] = tag, rank
                    rank += 1
            elif row == 0:
                rank = np.int64(0)
                for other in range(length):
                    for item in range(starts[other], stops[other]):
                        tags[item], index[item] = tag, rank
                        rank += 1
            scratch[row] = stops[row]
            return np.int8(0)

        self.prepare, self.prepare_small = prepare, prepare_small
        self.check, self.fill = check, fill
        self.iterator = CountingIterator(np.int64(0))
        self.discard = DiscardIterator()
        self.transforms = {}
        self.algorithms = {}
        self.initial = np.asarray(0, dtype=np.int64)

    def transform_small(self, stream):
        if "small" not in self.transforms:
            adapter = make_op_adapter(self.prepare_small)
            prepared = make_unary_transform(
                d_in=self.iterator, d_out=self.discard, op=adapter
            )
            self.transforms["small"] = adapter, prepared
        adapter, prepared = self.transforms["small"]
        prepared(
            d_in=self.iterator,
            d_out=self.discard,
            op=adapter,
            num_items=1,
            stream=stream,
        )

    def run(self, tags, index, scratch, counts):
        length = self.length
        # One arena per invocation; no input copy or scalar upload is needed.
        arena = cp.empty(7 * length + 1, dtype=cp.int64)
        starts, stops, positive, present_stops, positions, maxima, overlap = (
            arena[i * length : (i + 1) * length] for i in range(7)
        )
        serial = arena[-1:]
        arrays = (
            tags,
            index,
            scratch,
            counts,
            starts,
            stops,
            positive,
            present_stops,
            positions,
            maxima,
            serial,
        )
        for state, array in zip(self.states, arrays, strict=True):
            state.bind(array)
        stream = _Stream()
        try:
            if length <= 64:
                self.transform_small(stream)
            else:
                self.transform("prepare", self.prepare, self.discard, stream)
                for name, source, target, op in (
                    ("positions", positive, positions, OpKind.PLUS),
                    ("maxima", present_stops, maxima, OpKind.MAXIMUM),
                ):
                    build = {
                        "d_in": source,
                        "d_out": target,
                        "op": op,
                        "init_value": self.initial,
                    }
                    arguments = {**build, "num_items": length, "stream": stream}
                    self.algorithm(name, make_inclusive_scan, build, arguments)
                self.transform("check", self.check, overlap, stream)
                build = {
                    "d_in": overlap,
                    "d_out": serial,
                    "op": OpKind.MAXIMUM,
                    "h_init": self.initial,
                }
                arguments = {**build, "num_items": length, "stream": stream}
                self.algorithm("overlap", make_reduce_into, build, arguments)
            self.transform("fill", self.fill, self.discard, stream)
        finally:
            for state in self.states:
                state.clear()


@lru_cache(maxsize=32)
def _nested_setup(
    device, context, tag_dtype, index_dtype, starts_dtype, counts_dtype, tag, length
):
    return _NestedFill(tag_dtype, index_dtype, starts_dtype, counts_dtype, tag, length)


_nested_lock = RLock()


def nestedfill_tags_index(totags, toindex, tmpstarts, tag, fromcounts, length):
    if not length:
        return
    with _nested_lock:
        setup = _nested_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            totags.dtype.str,
            toindex.dtype.str,
            tmpstarts.dtype.str,
            fromcounts.dtype.str,
            int(tag),
            length,
        )
        setup.run(totags, toindex, tmpstarts, fromcounts)
