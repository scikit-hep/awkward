# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1.layout


def libs(*arrays):
    libs = set()

    def apply(layout, depth):
        if layout.identities is not None:
            libs.add(layout.identities.ptr_lib)

        if isinstance(layout, awkward1.layout.NumpyArray):
            libs.add(layout.ptr_lib)

        elif isinstance(layout, (
            awkward1.layout.ListArray32,
            awkward1.layout.ListArrayU32,
            awkward1.layout.ListArray64,
        )):
            libs.add(layout.starts.ptr_lib)
            libs.add(layout.stops.ptr_lib)

        elif isinstance(layout, (
            awkward1.layout.ListOffsetArray32,
            awkward1.layout.ListOffsetArrayU32,
            awkward1.layout.ListOffsetArray64,
        )):
            libs.add(layout.offsets.ptr_lib)

        elif isinstance(layout, (
            awkward1.layout.IndexedArray32,
            awkward1.layout.IndexedArrayU32,
            awkward1.layout.IndexedArray64,
            awkward1.layout.IndexedOptionArray32,
            awkward1.layout.IndexedOptionArray64,
        )):
            libs.add(layout.index.ptr_lib)

        elif isinstance(layout, (
            awkward1.layout.ByteMaskedArray,
            awkward1.layout.BitMaskedArray,
        )):
            libs.add(layout.mask.ptr_lib)

        elif isinstance(layout, (
            awkward1.layout.UnionArray8_32,
            awkward1.layout.UnionArray8_U32,
            awkward1.layout.UnionArray8_64,
        )):
            libs.add(layout.tags.ptr_lib)
            libs.add(layout.index.ptr_lib)

        elif isinstance(layout, awkward1.layout.VirtualArray):
            libs.add(layout.ptr_lib)

    for array in arrays:
        layout = awkward1.operations.convert.to_layout(
            array,
            allow_record=True,
            allow_other=True,
        )
        if isinstance(layout, (awkward1.layout.Content, awkward1.layout.Record)):
            awkward1._util.recursive_walk(layout, apply, materialize=False)
        elif isinstance(layout, numpy.ndarray):
            libs.add("cpu")
        elif type(layout).__module__.startswith("cupy."):
            libs.add("cuda")

    if libs == set() or libs == set(["cpu"]):
        return "cpu"
    elif libs == set(["cuda"]):
        return "cuda"
    else:
        return "mixed"


def of(*arrays):
    ptr_lib = libs(*arrays)

    if ptr_lib == "cpu":
        return Numpy.instance()
    elif ptr_lib == "cuda":
        return Cupy.instance()
    else:
        raise ValueError(
            """structure mixes 'cpu' and 'cuda' buffers; use one of

    ak.copy_to(array, 'cpu')
    ak.copy_to(array, 'cuda')

to obtain an unmixed array in main memory or the GPU."""
            + awkward1._util.exception_suffix(__file__))


class Singleton(object):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class NumpyMetadata(Singleton):
    bool = numpy.bool
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128

    intp = numpy.intp
    integer = numpy.integer
    floating = numpy.floating
    number = numpy.number
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    errstate = numpy.errstate
    newaxis = numpy.newaxis

    ndarray = numpy.ndarray

if hasattr(numpy, "float16"):
    NumpyMetadata.float16 = numpy.float16

if hasattr(numpy, "float128"):
    NumpyMetadata.float128 = numpy.float128

if hasattr(numpy, "complex256"):
    NumpyMetadata.complex256 = numpy.complex256

if hasattr(numpy, "datetime64"):
    NumpyMetadata.datetime64 = numpy.datetime64

if hasattr(numpy, "timedelta64"):
    NumpyMetadata.timedelta64 = numpy.timedelta64


class NumpyLike(Singleton):
    ############################ array creation

    def array(self, *args, **kwargs):
        # data[, dtype=[, copy=]]
        return self._module.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        # array[, dtype=]
        return self._module.asarray(*args, **kwargs)

    def frombuffer(self, *args, **kwargs):
        # array[, dtype=]
        return self._module.frombuffer(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        # shape/len[, dtype=]
        return self._module.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        # shape/len[, dtype=]
        return self._module.ones(*args, **kwargs)

    def empty(self, *args, **kwargs):
        # shape/len[, dtype=]
        return self._module.empty(*args, **kwargs)

    def full(self, *args, **kwargs):
        # shape/len, value[, dtype=]
        return self._module.full(*args, **kwargs)

    def arange(self, *args, **kwargs):
        # stop[, dtype=]
        # start, stop[, dtype=]
        # start, stop, step[, dtype=]
        return self._module.arange(*args, **kwargs)

    def meshgrid(self, *args, **kwargs):
        # *arrays, indexing="ij"
        return self._module.meshgrid(*args, **kwargs)

    ############################ testing

    def array_equal(self, *args, **kwargs):
        # array1, array2
        return self._module.array_equal(*args, **kwargs)

    def size(self, *args, **kwargs):
        # array
        return self._module.size(*args, **kwargs)

    def searchsorted(self, *args, **kwargs):
        # haystack, needle, side="right"
        return self._module.searchsorted(*args, **kwargs)

    ############################ manipulation

    def cumsum(self, *args, **kwargs):
        # arrays[, out=]
        return self._module.cumsum(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        # array
        return self._module.nonzero(*args, **kwargs)

    def unique(self, *args, **kwargs):
        # array
        return self._module.unique(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        # arrays
        return self._module.concatenate(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        # array, int
        # array1, array2
        return self._module.repeat(*args, **kwargs)

    def stack(self, *args, **kwargs):
        # arrays
        return self._module.stack(*args, **kwargs)

    def vstack(self, *args, **kwargs):
        # arrays
        return self._module.vstack(*args, **kwargs)

    def packbits(self, *args, **kwargs):
        # array
        return self._module.packbits(*args, **kwargs)

    def unpackbits(self, *args, **kwargs):
        # array
        return self._module.unpackbits(*args, **kwargs)

    def atleast_1d(self, *args, **kwargs):
        # *arrays
        return self._module.atleast_1d(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        # array, shape
        return self._module.broadcast_to(*args, **kwargs)

    ############################ ufuncs

    def sqrt(self, *args, **kwargs):
        # array
        return self._module.sqrt(*args, **kwargs)

    def exp(self, *args, **kwargs):
        # array
        return self._module.exp(*args, **kwargs)

    def true_divide(self, *args, **kwargs):
        # array1, array2
        return self._module.true_divide(*args, **kwargs)

    def bitwise_or(self, *args, **kwargs):
        # array1, array2[, out=output]
        return self._module.bitwise_or(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        # array1, array2
        return self._module.logical_and(*args, **kwargs)

    def equal(self, *args, **kwargs):
        # array1, array2
        return self._module.equal(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        # array
        return self._module.ceil(*args, **kwargs)

    ############################ reducers

    def all(self, *args, **kwargs):
        # array
        return self._module.all(*args, **kwargs)

    def any(self, *args, **kwargs):
        # array
        return self._module.any(*args, **kwargs)

    def count_nonzero(self, *args, **kwargs):
        # array
        return self._module.count_nonzero(*args, **kwargs)

    def sum(self, *args, **kwargs):
        # array
        return self._module.sum(*args, **kwargs)

    def prod(self, *args, **kwargs):
        # array
        return self._module.prod(*args, **kwargs)

    def min(self, *args, **kwargs):
        # array
        return self._module.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        # array
        return self._module.max(*args, **kwargs)

    def argmin(self, *args, **kwargs):
        # array[, axis=]
        return self._module.argmin(*args, **kwargs)

    def argmax(self, *args, **kwargs):
        # array[, axis=]
        return self._module.argmax(*args, **kwargs)


class Numpy(NumpyLike):
    def __init__(self):
        self._module = numpy

    @property
    def ma(self):
        return self._module.ma


class Cupy(NumpyLike):
    def __init__(self):
        try:
            import cupy
        except ImportError:
            raise ImportError(
            """to use CUDA arrays in Python, install the 'cupy' package with:

    pip install cupy --upgrade

or

    conda install cupy"""
        )
        self._module = cupy

    @property
    def ma(self):
        raise ValueError(
            "CUDA arrays cannot have missing values until CuPy implements "
            "numpy.ma.MaskedArray"
            + awkward1._util.exception_suffix(__file__)
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def asarray(self, array, dtype=None):
        if isinstance(array, (
            awkward1.highlevel.Array,
            awkward1.highlevel.Record,
            awkward1.layout.Content,
            awkward1.layout.Record,
        )):
            out = awkward1.operations.convert.to_cupy(array)
            if dtype is not None and out.dtype != dtype:
                return self._module.asarray(out, dtype=dtype)
            else:
                return out
        else:
            return self._module.asarray(array, dtype=dtype)

    def frombuffer(self, *args, **kwargs):
        np_array = numpy.frombuffer(*args, **kwargs)
        return self._module.array(np_array)

    def array_equal(self, array1, array2):
        if array1.shape != array2.shape:
            return False
        else:
            return self._module.all(array1 - array2 == 0)

    def repeat(self, array, repeats):
        if isinstance(repeats, self._module.ndarray):
            all_stops = self._module.cumsum(repeats)
            parents = self._module.zeros(all_stops[-1].item(), dtype=int)
            stops, stop_counts = self._module.unique(all_stops[:-1], return_counts=True)
            parents[stops] = stop_counts
            self._module.cumsum(parents, out=parents)
            return array[parents]
        else:
            return self._module.repeat(array, repeats)

    def all(self, array, axis=None):
        out = self._module.all(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def any(self, array, axis=None):
        out = self._module.any(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def count_nonzero(self, array, axis=None):
        out = self._module.count_nonzero(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def sum(self, array, axis=None):
        out = self._module.sum(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def prod(self, array, axis=None):
        out = self._module.prod(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def min(self, array, axis=None):
        out = self._module.min(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def max(self, array, axis=None):
        out = self._module.max(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def argmin(self, array, axis=None):
        out = self._module.argmin(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def argmax(self, array, axis=None):
        out = self._module.argmax(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out
