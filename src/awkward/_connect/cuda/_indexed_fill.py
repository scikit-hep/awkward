# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Prepared IndexedArray fill with an invocation-local runtime base."""

from functools import lru_cache
from threading import RLock

from cuda.compute import make_unary_transform
from cuda.compute.op import make_op_adapter

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cp = Cupy.instance()._module
np = Numpy.instance()._module
_lock = RLock()


class _Stream:
    def __init__(self):
        self.stream = cp.cuda.get_current_stream()

    def __cuda_stream__(self):
        return (0, self.stream.ptr)


class _BaseState:
    # CCCL turns this captured CUDA-array-interface proxy into a device array.
    # It retains metadata only, never an invocation's array owner.
    dtype = np.dtype(np.int64)

    def __init__(self):
        self.clear()

    def bind(self, array):
        self.__cuda_array_interface__ = dict(array.__cuda_array_interface__)

    def clear(self):
        self.__cuda_array_interface__ = {
            "shape": (1,),
            "strides": None,
            "typestr": self.dtype.str,
            "data": (0, False),
            "version": 3,
        }


class _PreparedFill:
    def __init__(self, output_dtype):
        self.base = _BaseState()
        base = self.base
        out_dtype = np.dtype(output_dtype).type

        def fill(value):
            return out_dtype(-1) if value < 0 else out_dtype(value + base[0])

        self.operation = make_op_adapter(fill)
        self.transform = None

    def __call__(self, source, output, base):
        # Keep the parameter array alive until the launch is enqueued. CuPy's
        # stream-local allocator orders reuse after the dependent device work.
        # Each invocation owns its parameter storage, including concurrent streams.
        parameter = cp.asarray([base], dtype=cp.int64)
        self.base.bind(parameter)
        try:
            if self.transform is None:
                self.transform = make_unary_transform(
                    d_in=source, d_out=output, op=self.operation
                )
            self.transform(
                d_in=source,
                d_out=output,
                op=self.operation,
                num_items=source.size,
                stream=_Stream(),
            )
        finally:
            self.base.clear()


@lru_cache(maxsize=32)
def _setup(device, context, input_dtype, output_dtype):
    return _PreparedFill(output_dtype)


def fill(toindex, offset, fromindex, length, base):
    if length == 0:
        return
    # Protect host descriptor rebinding/launch, without waiting for the GPU.
    with _lock:
        prepared = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            fromindex.dtype.str,
            toindex.dtype.str,
        )
        prepared(fromindex[:length], toindex[offset : offset + length], base)
