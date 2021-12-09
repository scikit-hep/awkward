# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

import numpy

import awkward.nplike
import awkward._util
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class NoError(object):
    str = None
    filename = None
    pass_through = False
    attempt = ak._util.kSliceNone
    id = ak._util.kSliceNone


class NoKernel(object):
    def __call__(self, *args):
        return NoError()


class UnknownLengthType(object):
    def __repr__(self):
        return "UnknownLength"

    def __str__(self):
        return "??"

    def __eq__(self, other):
        return isinstance(other, UnknownLengthType)

    def __add__(self, other):
        return UnknownLength

    def __radd__(self, other):
        return UnknownLength

    def __sub__(self, other):
        return UnknownLength

    def __rsub__(self, other):
        return UnknownLength


UnknownLength = UnknownLengthType()


class TypeTracerArray(object):
    @classmethod
    def from_array(cls, array, dtype=None):
        if isinstance(array, ak._v2.index.Index):
            array = array.data

        if dtype is None:
            dtype = array.dtype

        shape = list(array.shape)
        shape[0] = UnknownLength

        return cls(dtype, shape=shape)

    def __init__(self, dtype, shape=None):
        self._dtype = np.dtype(dtype)
        self.shape = shape

    def __repr__(self):
        dtype = repr(self._dtype)
        shape = ""
        if self._shape != (UnknownLength,):
            shape = ", " + repr(self._shape)
        return "TypeTracerArray({0}{1})".format(dtype, shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value is None or isinstance(value, (numbers.Integral, UnknownLengthType)):
            value = (UnknownLength,)
        elif len(value) == 0:
            value = ()
        else:
            value = (UnknownLength,) + tuple(value[1:])
        self._shape = value

    @property
    def strides(self):
        out = (self._dtype.itemsize,)
        for x in self._shape[:0:-1]:
            out = (x * out[0],) + out
        return out

    @property
    def nplike(self):
        return TypeTracer.instance()

    @property
    def ndim(self):
        return len(self._shape)

    def __iter__(self):
        raise AssertionError(
            "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
        )

    def __array__(self, *args, **kwargs):
        raise AssertionError(
            "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
        )

    def itemsize(self):
        return self._dtype.itemsize

    class _CTypes(object):
        data = 0

    @property
    def ctypes(self):
        return self._CTypes

    def __len__(self):
        raise AssertionError("bug in Awkward Array: attempt to get length of a TypeTracerArray")

    def __setitem__(self, where, what):
        raise AssertionError("bug in Awkward Array: attempt to set values of a TypeTracerArray")

    def __getitem__(self, where):
        raise AssertionError("bug in Awkward Array: attempt to get values from a TypeTracerArray")

    def __lt__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape, False, False)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape, False, False)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape, False, False)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape, False, False)
        else:
            return NotImplemented

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        assert len(args) != 0
        assert all(isinstance(x, numbers.Integral) for x in args)
        assert all(x >= 0 for x in args[1:])

        return TypeTracerArray(self._dtype, (UnknownLength,) + args[1:])

    def copy(self):
        return self


unset = object()


class TypeTracer(ak.nplike.NumpyLike):
    known_data = False
    known_shape = False

    def to_rectilinear(self, array, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, name_and_types):
        return NoKernel()

    @property
    def ma(self):
        raise NotImplementedError

    @property
    def char(self):
        raise NotImplementedError

    @property
    def ndarray(self):
        return TypeTracerArray

    ############################ array creation

    def array(self, data, dtype=unset, **kwargs):
        # data[, dtype=[, copy=]]
        if dtype is unset:
            dtype = data.dtype
        return TypeTracerArray.from_array(data, dtype=dtype)

    def asarray(self, array, dtype=unset, **kwargs):
        # array[, dtype=][, order=]
        if dtype is unset:
            dtype = array.dtype
        return TypeTracerArray.from_array(array, dtype=dtype)

    def ascontiguousarray(self, array, dtype=unset, **kwargs):
        # array[, dtype=]
        if dtype is unset:
            dtype = array.dtype
        return TypeTracerArray.from_array(array, dtype=dtype)

    def isscalar(self, *args, **kwargs):
        raise NotImplementedError

    def frombuffer(self, *args, **kwargs):
        # array[, dtype=]
        raise NotImplementedError

    def zeros(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def ones(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def empty(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def full(self, shape, value, dtype=unset, **kwargs):
        # shape/len, value[, dtype=]
        if dtype is unset:
            dtype = numpy.array(value).dtype
        return TypeTracerArray(dtype, shape, fill_zero=value, fill_other=value)

    def zeros_like(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def ones_like(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def full_like(self, *args, **kwargs):
        # array, fill_value
        raise NotImplementedError

    def arange(self, *args, **kwargs):
        # stop[, dtype=]
        # start, stop[, dtype=]
        # start, stop, step[, dtype=]
        raise NotImplementedError

    def meshgrid(self, *args, **kwargs):
        # *arrays, indexing="ij"
        raise NotImplementedError

    ############################ testing

    def shape(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def array_equal(self, *args, **kwargs):
        # array1, array2
        raise NotImplementedError

    def size(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def searchsorted(self, *args, **kwargs):
        # haystack, needle, side="right"
        raise NotImplementedError

    def argsort(self, *args, **kwargs):
        # array
        raise NotImplementedError

    ############################ manipulation

    def broadcast_arrays(self, *args, **kwargs):
        # array1[, array2[, ...]]
        raise NotImplementedError

    def add(self, *args, **kwargs):
        # array1, array2[, out=]
        raise NotImplementedError

    def cumsum(self, *args, **kwargs):
        # arrays[, out=]
        raise NotImplementedError

    def cumprod(self, *args, **kwargs):
        # arrays[, out=]
        raise NotImplementedError

    def nonzero(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def unique(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def concatenate(self, arrays):
        return TypeTracerArray(arrays[0].dtype, (UnknownLength,) + arrays[0].shape[1:])

    def repeat(self, *args, **kwargs):
        # array, int
        # array1, array2
        raise NotImplementedError

    def stack(self, *args, **kwargs):
        # arrays
        raise NotImplementedError

    def vstack(self, *args, **kwargs):
        # arrays
        raise NotImplementedError

    def packbits(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def unpackbits(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def atleast_1d(self, *args, **kwargs):
        # *arrays
        raise NotImplementedError

    def broadcast_to(self, *args, **kwargs):
        # array, shape
        raise NotImplementedError

    def append(self, *args, **kwargs):
        # array, element
        raise NotImplementedError

    def where(self, *args, **kwargs):
        # array, element
        raise NotImplementedError

    ############################ ufuncs

    def sqrt(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def exp(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def true_divide(self, *args, **kwargs):
        # array1, array2
        raise NotImplementedError

    def bitwise_or(self, *args, **kwargs):
        # array1, array2[, out=output]
        raise NotImplementedError

    def logical_and(self, *args, **kwargs):
        # array1, array2
        raise NotImplementedError

    def equal(self, *args, **kwargs):
        # array1, array2
        raise NotImplementedError

    def ceil(self, *args, **kwargs):
        # array
        raise NotImplementedError

    ############################ almost-ufuncs

    def nan_to_num(self, *args, **kwargs):
        # array, copy=True, nan=0.0, posinf=None, neginf=None
        raise NotImplementedError

    def isclose(self, *args, **kwargs):
        # a, b, rtol=1e-05, atol=1e-08, equal_nan=False
        raise NotImplementedError

    ############################ reducers

    def all(self, array, prefer):
        # array
        return prefer

    def any(self, array, prefer):
        # array
        return prefer

    def count_nonzero(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def sum(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def prod(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def min(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def max(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def argmin(self, *args, **kwargs):
        # array[, axis=]
        raise NotImplementedError

    def argmax(self, *args, **kwargs):
        # array[, axis=]
        raise NotImplementedError

    def array_str(
        self, array, max_line_width=unset, precision=unset, suppress_small=unset
    ):
        # array, max_line_width, precision=None, suppress_small=None
        return "[?? ... ??]"

    def datetime_as_string(self, *args, **kwargs):
        raise NotImplementedError
