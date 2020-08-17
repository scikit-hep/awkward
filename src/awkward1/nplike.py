# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy


def of(*arrays):
    return Numpy.instance()


class NumpyLike(object):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class NumpyMetadata(NumpyLike):
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
    NumpyLike.float16 = numpy.float16

if hasattr(numpy, "float128"):
    NumpyLike.float128 = numpy.float128

if hasattr(numpy, "complex256"):
    NumpyLike.complex256 = numpy.complex256

if hasattr(numpy, "datetime64"):
    NumpyLike.datetime64 = numpy.datetime64

if hasattr(numpy, "timedelta64"):
    NumpyLike.timedelta64 = numpy.timedelta64


class Numpy(NumpyLike):
    def __init__(self):
        self._module = numpy

    ############################ submodules

    @property
    def ma(self):
        return self._module.ma

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
        # stops, where, side="right"
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
