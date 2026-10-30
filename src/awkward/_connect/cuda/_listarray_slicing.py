# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Prepared jagged application and output-parallel range slicing."""

from functools import lru_cache

from cuda.compute import OpKind, make_inclusive_scan
from cuda.compute.op import make_op_adapter

from awkward._connect.cuda._listarray_prepared import _ArrayState, _lock, _Prepared
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module


class _Scan(_Prepared):
    def __init__(self, length):
        super().__init__(length, OpKind.MINIMUM, length)
        self.scanner = None
        self.scan_operation = make_op_adapter(OpKind.PLUS)
        self.zero = np.asarray(0, dtype=np.int64)
        self.scan_workspace_sizes = {}

    def configure(self, length, parameters):
        self.length = length
        self.chunks = (length + 63) // 64
        self.initial = np.asarray(length, dtype=np.int64)
        self.runtime_values = (length, *parameters)

    def scan(self, counts, output, stream):
        if self.scanner is None:
            self.scanner = make_inclusive_scan(
                d_in=counts, d_out=output, op=self.scan_operation, init_value=self.zero
            )
        arguments = {
            "d_in": counts,
            "d_out": output,
            "num_items": self.length,
            "op": self.scan_operation,
            "init_value": self.zero,
            "stream": stream,
        }
        if self.length not in self.scan_workspace_sizes:
            if len(self.scan_workspace_sizes) >= 32:
                self.scan_workspace_sizes.pop(next(iter(self.scan_workspace_sizes)))
            self.scan_workspace_sizes[self.length] = self.scanner(
                temp_storage=None, **arguments
            )
        storage_bytes = self.scan_workspace_sizes[self.length]
        storage = cp.empty(storage_bytes, dtype=cp.uint8)
        self.scanner(temp_storage=storage, **arguments)


class _JaggedApply(_Scan):
    def __init__(self, dtypes, length, inner_length, content_length):
        super().__init__(length)
        self.states = tuple(
            _ArrayState(dtype) for dtype in (*dtypes, *([np.int64] * 4))
        )
        (
            offsets,
            carry,
            ss,
            se,
            indices,
            starts,
            stops,
            counts,
            positions,
            first,
            status,
        ) = self.states

        runtime = self.runtime
        self.runtime_values = (
            length,
            inner_length,
            content_length,
        )

        def check(chunk):
            boundary = np.int64(runtime[0])
            for row in range(chunk * 64, min((chunk + 1) * 64, runtime[0])):
                # Every row is initialized even after this chunk's first error:
                # the scan must never consume undefined scratch values.
                counts[row] = 0
                if row == 0:
                    positions[0] = 0
                if row < boundary:
                    left, right = ss[row], se[row]
                    # Empty slices skip all structural checks in the reference.
                    if left != right:
                        if right < left or right > runtime[1]:
                            boundary = row
                        else:
                            start, stop = np.int64(starts[row]), np.int64(stops[row])
                            if stop < start or (start != stop and stop > runtime[2]):
                                boundary = row
                            else:
                                size = stop - start
                                for j in range(left, right):
                                    index = indices[j]
                                    if index < -size or index >= size:
                                        boundary = row
                                        break
                                    counts[row] += 1
            return boundary

        def write(row):
            boundary = first[0]
            if row <= boundary:
                offsets[row] = positions[row]
                if counts[row] > 0:
                    start = np.int64(starts[row])
                    size = np.int64(stops[row]) - start
                    for j in range(counts[row]):
                        index = indices[ss[row] + j]
                        if index < 0:
                            index += size
                        carry[positions[row] + j] = start + index
            if row == 0:
                status[0] = -1
                status[1] = np.int64(9223372036854775807)
                if boundary == runtime[0]:
                    offsets[runtime[0]] = positions[runtime[0]]
                else:
                    left, right = ss[boundary], se[boundary]
                    code = np.int64(4)
                    if right < left:
                        code = np.int64(0)
                    elif right > runtime[1]:
                        code = np.int64(1)
                        status[1] = right
                    else:
                        start, stop = (
                            np.int64(starts[boundary]),
                            np.int64(stops[boundary]),
                        )
                        if stop < start:
                            code = np.int64(2)
                        elif start != stop and stop > runtime[2]:
                            code = np.int64(3)
                        else:
                            status[1] = indices[left + counts[boundary]]
                    status[0] = boundary * 5 + code
            return np.int8(0)

        self.check, self.write = check, write

    def __call__(self, arrays, stream):
        counts = cp.empty(self.length, dtype=cp.int64)
        # Scan into scratch: offsets beyond the earliest failing row must stay
        # untouched, including the terminal offset on error.
        positions = cp.empty(self.length + 1, dtype=cp.int64)
        first = cp.empty(1, dtype=cp.int64)
        status = cp.empty(2, dtype=cp.int64)
        self.bind((*arrays, counts, positions, first, status))
        try:
            self.find(first, stream)
            self.scan(counts, positions[1:], stream)
            self.transform("write", self.write, self.discard, self.length, stream)
        finally:
            self.clear()
        return status


class _NextRange(_Scan):
    def __init__(self, dtypes, length, start, stop, step):
        super().__init__(length)
        self.states = tuple(_ArrayState(dtype) for dtype in (*dtypes, np.int64))
        offsets, carry, starts, stops, positions = self.states
        narrow = np.dtype(dtypes[0]) != np.dtype(np.int64)

        runtime = self.runtime
        self.runtime_values = (
            length,
            start,
            stop,
            step,
        )

        def count(row):
            if row == 0:
                positions[0] = 0
            stride = np.int64(runtime[3])
            size = np.int64(stops[row]) - np.int64(starts[row])
            left, right = np.int64(runtime[1]), np.int64(runtime[2])
            if stride > 0:
                if left == 9223372036854775807:
                    left = 0
                elif left < 0:
                    left += size
                left = min(max(left, 0), size)
                if right == 9223372036854775807:
                    right = size
                elif right < 0:
                    right += size
                right = min(max(right, 0), size)
                right = max(right, left)
                return (
                    np.int64(0) if right <= left else 1 + (right - left - 1) // stride
                )
            if left == 9223372036854775807:
                left = size - 1
            elif left < 0:
                left += size
            left = min(max(left, -1), size - 1)
            if right == 9223372036854775807:
                right = -1
            elif right < 0:
                right += size
            right = min(max(right, -1), size - 1)
            right = min(right, left)
            if left <= right:
                return np.int64(0)
            if stride == -9223372036854775807 - 1:
                return np.int64(1)
            return 1 + (left - right - 1) // -stride

        def write(k):
            if narrow and k <= runtime[0]:
                offsets[k] = positions[k]
            if k < positions[runtime[0]]:
                # Prefix offsets are monotone, including repeated empty lists.
                left, right = 0, runtime[0] + 1
                while left < right:
                    middle = (left + right) // 2
                    if positions[middle] <= k:
                        left = middle + 1
                    else:
                        right = middle
                row = left - 1
                size = np.int64(stops[row]) - np.int64(starts[row])
                begin = np.int64(runtime[1])
                if runtime[3] > 0:
                    if begin == 9223372036854775807:
                        begin = 0
                    elif begin < 0:
                        begin += size
                    begin = min(max(begin, 0), size)
                else:
                    if begin == 9223372036854775807:
                        begin = size - 1
                    elif begin < 0:
                        begin += size
                    begin = min(max(begin, -1), size - 1)
                carry[k] = (
                    np.int64(starts[row])
                    + begin
                    + (k - positions[row]) * np.int64(runtime[3])
                )
            return np.int8(0)

        self.count, self.write = count, write

    def __call__(self, arrays, stream):
        counts = cp.empty(self.length, dtype=cp.int64)
        narrow = arrays[0].dtype != np.dtype(np.int64)
        # Carry generation uses full-width prefixes even if output offsets
        # narrow on assignment. Reuse the output directly for int64 offsets.
        positions = cp.empty(self.length + 1, dtype=cp.int64) if narrow else arrays[0]
        self.bind((*arrays, positions))
        try:
            self.transform("count", self.count, counts, self.length, stream)
            self.scan(counts, positions[1 : self.length + 1], stream)
            items = max(arrays[1].size, self.length + 1) if narrow else arrays[1].size
            if items:
                self.transform("write", self.write, self.discard, items, stream)
        finally:
            self.clear()


@lru_cache(maxsize=32)
def _setup(device, context, kind, dtypes):
    if kind == "apply":
        return _JaggedApply(dtypes, 0, 0, 0)
    return _NextRange(dtypes, 0, 0, 0, 1)


def run(kind, arrays, length, parameters, stream):
    with _lock:
        setup = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            kind,
            tuple(array.dtype.str for array in arrays),
        )
        setup.configure(length, parameters)
        return setup(arrays, stream)
