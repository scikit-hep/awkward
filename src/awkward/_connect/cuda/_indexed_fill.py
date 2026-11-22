# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Prepared IndexedArray fill with stream-ordered reusable runtime state."""

from functools import lru_cache
from threading import RLock

from cuda.compute import make_unary_transform
from cuda.compute.op import make_op_adapter

from awkward._connect.cuda._compute_cache import load_or_build
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module


class _PreparedFill:
    def __init__(self, output_dtype):
        self.lock = RLock()
        with cp.cuda.Stream.null:
            self.base = cp.empty(1, dtype=cp.int64)
        self.last_base = None
        base = self.base
        out_dtype = np.dtype(output_dtype).type

        def fill(value):
            return out_dtype(-1) if value < 0 else out_dtype(value + base[0])

        self.operation = make_op_adapter(fill)
        self.transform = None

    def __call__(self, source, output, base):
        # Both parameter updates and transforms use the legacy default stream.
        # The lock keeps each update/transform pair together in its queue, so
        # later calls cannot overwrite the parameter before earlier calls read it.
        # This also holds when a non-blocking CuPy stream is current.
        with self.lock, cp.cuda.Stream.null:
            try:
                base = np.int64(base)
                if self.last_base is None or base != self.last_base:
                    # CuPy passes the scalar as a kernel argument (or uses an
                    # async memset for zero), without a host-to-device upload.
                    self.base.fill(base)
                    self.last_base = base
                if self.transform is None:
                    self.transform = load_or_build(
                        make_unary_transform,
                        source_path=__file__,
                        d_in=source,
                        d_out=output,
                        op=self.operation,
                    )
                self.transform(
                    d_in=source,
                    d_out=output,
                    op=self.operation,
                    num_items=source.size,
                    stream=None,
                )
            except Exception:
                self.last_base = None
                raise


# Only lookup/construction is serialized here, not compilation or submission.
# lru_cache alone can construct duplicate objects on concurrent cold misses.
_setup_lock = RLock()


@lru_cache(maxsize=32)
def _setup(device, context, input_dtype, output_dtype):
    return _PreparedFill(output_dtype)


def fill(toindex, offset, fromindex, length, base):
    if length == 0:
        return
    device = cp.cuda.runtime.getDevice()
    context = cp.cuda.driver.ctxGetCurrent()
    with _setup_lock:
        prepared = _setup(device, context, fromindex.dtype.str, toindex.dtype.str)
    prepared(fromindex[:length], toindex[offset : offset + length], base)
