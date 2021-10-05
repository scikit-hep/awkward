# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers

import awkward.nplike
import awkward._util
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class NoError(object):
    str = None
    filename = None
    pass_through = False
    attempt = ak._util.kSliceNone
    id = ak._util.kSliceNone


class NoKernel(object):
    def __call__(self, *args):
        return NoError()


class Interval(object):
    @classmethod
    def unknown(cls):
        return cls(0, None)

    @classmethod
    def exact(cls, value):
        return cls(value, value)

    def __init__(self, min, max):
        assert max is None or min <= max
        self._min = min
        self._max = max

    def __repr__(self):
        return "Interval(min={0}, max={1})".format(self._min, self._max)

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def __index__(self):
        return self._min

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self._min == other._min and self._max == other._max

        elif isinstance(other, numbers.Integral):
            if self._max is None:
                return self._min <= other
            else:
                return self._min <= other <= self._max

    def __add__(self, other):
        if isinstance(other, Interval):
            if self._max is None or other._max is None:
                return Interval(self._min + other._min, None)
            else:
                return Interval(self._min + other._min, self._max + other._max)

        elif isinstance(other, numbers.Integral):
            if self._max is None:
                return Interval(self._min + other, None)
            else:
                return Interval(self._min + other, self._max + other)

        else:
            raise TypeError("cannot add Interval and {0}".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self._min += other
        if self._max is not None:
            self._max += other
        return self


class TypeTracerArray(object):
    def __init__(self, dtype, shape=None, fill=0):
        if shape is None:
            shape = (Interval.unknown(),)
        elif isinstance(shape, numbers.Integral):
            shape = (shape,)

        self._dtype = np.dtype(dtype)
        self._shape = shape
        self._fill = fill

    def __repr__(self):
        dtype = repr(self._dtype)

        shape = ""
        if self._shape != (Interval.unknown(),):
            shape = ", " + repr(self._shape)

        fill = ""
        if self._fill != 0:
            fill = ", " + repr(self._fill)

        return "TypeTracerArray({0}{1}{2})".format(dtype, shape, fill)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill

    @property
    def ndim(self):
        return len(self._shape)

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, where):
        if len(self._shape) == 1:
            return self._dtype.type(self._fill)
        else:
            return numpy.full(self._shape[1:], self._fill, dtype=self._dtype)


class TypeTracer(ak.nplike.NumpyLike):
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
