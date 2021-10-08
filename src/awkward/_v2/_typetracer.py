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


class Interval(object):
    @classmethod
    def unknown(cls):
        return cls(0, None)

    @classmethod
    def exact(cls, value):
        return cls(value, value)

    def __init__(self, min, max):
        assert max is None or min <= max
        self._min = int(min)
        self._max = None if max is None else int(max)

    def __str__(self):
        if self._max is None:
            return "{0}...".format(self._min)
        else:
            return "{0}...{1}".format(self._min, self._max)

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

    def __ge__(self, other):
        if isinstance(other, Interval):
            return NotImplemented

        elif isinstance(other, numbers.Integral):
            return self._min >= other

        else:
            return NotImplemented


def _length_after_slice(slice, original_length):
    start, stop, step = slice.indices(original_length)
    assert step != 0

    if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
        d, m = divmod(abs(start - stop), abs(step))
        return d + (1 if m != 0 else 0)
    else:
        return 0


class TypeTracerArray(object):
    @classmethod
    def from_array(cls, array, dtype=None, fill=0):
        if dtype is None:
            dtype = array.dtype

        shape = list(array.shape)
        if len(shape) != 0 and not isinstance(shape[0], Interval):
            shape[0] = Interval.exact(shape[0])

        return cls(dtype, shape=shape, fill=fill)

    def __init__(self, dtype, shape=None, fill=0):
        if shape is None:
            shape = (Interval.unknown(),)
        elif isinstance(shape, Interval):
            shape = (shape,)
        elif isinstance(shape, numbers.Integral):
            shape = (Interval.exact(shape),)
        else:
            if len(shape) == 0:
                shape = ()
            elif isinstance(shape[0], Interval):
                if not isinstance(shape, tuple):
                    shape = tuple(shape)
            elif isinstance(shape[0], numbers.Integral):
                shape = (Interval.exact(shape[0]),) + tuple(shape[1:])

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
    def nplike(self):
        return TypeTracer.instance()

    @property
    def ndim(self):
        return len(self._shape)

    def __iter__(self):
        raise AssertionError(
            "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
        )

    def __array__(self):
        return self.__iter__()

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, where):
        if isinstance(where, tuple):
            try:
                i = where.index(Ellipsis)
            except ValueError:
                pass
            else:
                before, after = where[:i], where[i + 1 :]
                missing = max(0, len(self._shape) - (len(before) + len(after)))
                where = before + (slice(None, None, None),) * missing + after

        if isinstance(where, numbers.Integral):
            if len(self._shape) == 1:
                return self._dtype.type(self._fill)
            else:
                return numpy.full(self._shape[1:], self._fill, dtype=self._dtype)

        elif isinstance(where, slice):
            length1 = _length_after_slice(where, self._shape[0].min)

            if self._shape[0].max is not None:
                length2 = _length_after_slice(where, self._shape[0].max)
            else:
                if where.start is not None and where.stop is not None:
                    length2 = _length_after_slice(
                        where, max(abs(where.start), abs(where.stop))
                    )
                elif where.stop is not None:
                    length2 = _length_after_slice(abs(where.stop))
                else:
                    length2 = None

            shape = (Interval(length1, length2),) + self._shape[1:]

            return TypeTracerArray(self._dtype, shape=shape, fill=self._fill)

        elif (
            hasattr(where, "dtype")
            and hasattr(where, "shape")
            and issubclass(where.dtype.type, np.integer)
        ):
            assert len(self._shape) != 0
            shape = where.shape + self._shape[1:]
            return TypeTracerArray(self._dtype, shape=shape, fill=self._fill)

        elif isinstance(where, tuple) and any(
            hasattr(x, "dtype") and hasattr(x, "shape") for x in where
        ):
            for num_basic, wh in enumerate(where):  # noqa: B007
                if not isinstance(wh, slice):
                    break

            if num_basic != 0:
                tmp = self.__getitem__(where[:num_basic])
                basic_shape = tmp._shape[:num_basic]
            else:
                basic_shape = ()

            shapes = []
            for j in range(num_basic, len(where)):
                wh = where[j]
                if isinstance(wh, numbers.Integral):
                    shapes.append(numpy.array(0))
                elif hasattr(wh, "dtype") and hasattr(wh, "shape"):
                    sh = [
                        x.min if isinstance(x, Interval) else int(x) for x in wh.shape
                    ]
                    shapes.append(
                        numpy.lib.stride_tricks.as_strided(
                            numpy.array(0), shape=sh, strides=[0] * len(sh)
                        )
                    )
                else:
                    raise NotImplementedError(repr(wh))

            slicer_shape = numpy.broadcast_arrays(*shapes)[0].shape

            shape = basic_shape + slicer_shape + self._shape[num_basic + len(shapes) :]
            if len(shape) == 0 or isinstance(shape[0], Interval):
                fixed_shape = shape
            else:
                fixed_shape = (Interval.exact(shape[0]),) + shape[1:]

            return TypeTracerArray(self._dtype, shape=fixed_shape, fill=self._fill)

        elif (
            isinstance(where, tuple)
            and len(where) > 0
            and isinstance(where[0], (numbers.Integral, slice))
        ):
            head, tail = where[0], where[1:]
            next = self.__getitem__(head)

            inner_shape = next.shape[1:]
            after_shape = []
            for i, wh in enumerate(tail):
                if isinstance(wh, int):
                    pass
                elif isinstance(wh, slice):
                    after_shape.append(_length_after_slice(wh, inner_shape[i]))
                else:
                    raise NotImplementedError(repr(wh))

            shape = (next._shape[0],) + tuple(after_shape)
            return TypeTracerArray(self._dtype, shape=shape, fill=self._fill)

        else:
            raise NotImplementedError(repr(where))

    def __lt__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, shape=self._shape, fill=False)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, shape=self._shape, fill=False)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, shape=self._shape, fill=False)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, shape=self._shape, fill=False)
        else:
            return NotImplemented

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        assert len(args) != 0
        assert all(isinstance(x, numbers.Integral) for x in args)
        assert all(x >= 0 for x in args[1:])

        if args[0] < 0:
            minitems, maxitems = self._shape[0].min, self._shape[0].max
            for x in self._shape[1:]:
                minitems *= x
                if maxitems is not None:
                    maxitems *= x

            divisor = 1
            for x in args[1:]:
                divisor *= x

            minlength = minitems // divisor
            if maxitems is None:
                maxlength = None
            else:
                maxlength = maxitems // divisor

            shape = (Interval(minlength, maxlength),) + args[1:]

        else:
            shape = (Interval.exact(args[0]),) + args[1:]

        return TypeTracerArray(self._dtype, shape=shape, fill=self._fill)

    def copy(self):
        return self


unset = object()


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

    def zeros(self, *args, **kwargs):
        # shape/len[, dtype=]
        raise NotImplementedError

    def ones(self, *args, **kwargs):
        # shape/len[, dtype=]
        raise NotImplementedError

    def empty(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def full(self, *args, **kwargs):
        # shape/len, value[, dtype=]
        raise NotImplementedError

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

    def concatenate(self, *args, **kwargs):
        # arrays
        raise NotImplementedError

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
