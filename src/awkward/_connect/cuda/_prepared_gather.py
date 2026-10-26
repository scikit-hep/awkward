# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Reusable CCCL offset transforms, without owning caller buffers."""

from functools import lru_cache
from threading import RLock

import numpy as np
from cuda.compute import CountingIterator, make_unary_transform
from cuda.compute.op import make_op_adapter

from awkward._nplikes.cupy import Cupy

cp = Cupy.instance()._module
# CCCL prepared transforms mutate their launch descriptors. Serialize host setup
# and launch, including factories that may share a globally cached descriptor.
_lock = RLock()


class _ArrayState:
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


class _Gather:
    def __init__(self, outer_dtype, inner_dtype):
        self.outer = outer = _ArrayState(outer_dtype)
        self.inner = inner = _ArrayState(inner_dtype)

        def gather(i):
            return inner[outer[i]]

        self.operation = None
        self.gather = gather
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None

    def __call__(self, output, outer, inner, length, stream):
        self.outer.bind(outer)
        self.inner.bind(inner)
        try:
            if self.operation is None:
                self.operation = make_op_adapter(self.gather)
            if self.prepared is None:
                self.prepared = make_unary_transform(
                    d_in=self.iterator, d_out=output, op=self.operation
                )
            self.prepared(
                d_in=self.iterator,
                d_out=output,
                op=self.operation,
                num_items=length,
                stream=stream,
            )
        finally:
            # Kernel arguments have been copied at launch. Retain no array
            # owners or active pointers in the Python operator's captured state.
            self.outer.clear()
            self.inner.clear()


@lru_cache(maxsize=16)
def _setup(device, context, outer_dtype, inner_dtype, output_dtype):
    return _Gather(outer_dtype, inner_dtype)


def gather_offsets(output, outer, inner, length, stream):
    if not length:
        return
    with _lock:
        setup = _setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            outer.dtype.str,
            inner.dtype.str,
            output.dtype.str,
        )
        setup(output, outer, inner, length, stream)


class _Ranges(_Gather):
    # Width is host metadata and part of the setup key. The backward search
    # avoids materializing one position per element or launching a reduction.
    def __init__(self, starts_dtype, source_dtype, width):
        self.outer = starts = _ArrayState(starts_dtype)
        self.inner = source = _ArrayState(source_dtype)

        def find_stop(i):
            start = i * width
            stop = start + width
            while stop > start and source[stop - 1] == -1:
                stop -= 1
            starts[i] = start
            return stop

        self.operation = None
        self.gather = find_stop
        self.iterator = CountingIterator(np.int64(0))
        self.prepared = None


@lru_cache(maxsize=16)
def _ranges_setup(device, context, starts_dtype, source_dtype, stops_dtype, width):
    return _Ranges(starts_dtype, source_dtype, width)


def nonlocal_offsets(starts, stops, source, length, groups, stream):
    with _lock:
        setup = _ranges_setup(
            cp.cuda.runtime.getDevice(),
            cp.cuda.driver.ctxGetCurrent(),
            starts.dtype.str,
            source.dtype.str,
            stops.dtype.str,
            length // groups,
        )
        setup(stops, starts, source, groups, stream)
