# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers

import numpy

import awkward.nplike
import awkward._util
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class NoError:
    str = None
    filename = None
    pass_through = False
    attempt = ak._util.kSliceNone
    id = ak._util.kSliceNone


class NoKernel:
    def __call__(self, *args):
        return NoError()


class UnknownLengthType:
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

    def __mul__(self, other):
        return UnknownLength

    def __rmul__(self, other):
        return UnknownLength

    def __truediv__(self, other):
        return UnknownLength

    def __floordiv__(self, other):
        return UnknownLength

    def __rdiv__(self, other):
        return UnknownLength

    def __rfloordiv__(self, other):
        return UnknownLength

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False


UnknownLength = UnknownLengthType()


def _emptyarray(x):
    if isinstance(x, UnknownScalar):
        return numpy.empty(0, x._dtype)
    elif hasattr(x, "dtype"):
        return numpy.empty(0, x.dtype)
    else:
        return numpy.empty(0, numpy.array(x).dtype)


class UnknownScalar:
    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"UnknownScalar({self._dtype!r})"

    def __str__(self):
        return f"unknown-{str(self._dtype)}"

    def __eq__(self, other):
        return isinstance(other, UnknownScalar) and self._dtype == other._dtype

    def __add__(self, other):
        return UnknownScalar((_emptyarray(self) + _emptyarray(other)).dtype)

    def __radd__(self, other):
        return UnknownScalar((_emptyarray(self) + _emptyarray(other)).dtype)

    def __sub__(self, other):
        return UnknownScalar((_emptyarray(self) - _emptyarray(other)).dtype)

    def __rsub__(self, other):
        return UnknownScalar((_emptyarray(self) - _emptyarray(other)).dtype)

    def __mul__(self, other):
        return UnknownScalar((_emptyarray(self) * _emptyarray(other)).dtype)

    def __rmul__(self, other):
        return UnknownScalar((_emptyarray(self) * _emptyarray(other)).dtype)

    def __truediv__(self, other):
        return UnknownScalar((_emptyarray(self) / _emptyarray(other)).dtype)

    def __floordiv__(self, other):
        return UnknownScalar((_emptyarray(self) // _emptyarray(other)).dtype)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False


class MaybeNone:
    def __init__(self, content):
        self._content = content

    @property
    def content(self):
        return self._content

    def __eq__(self, other):
        if isinstance(other, MaybeNone):
            return self._content == other._content
        else:
            return False

    def __repr__(self):
        return f"MaybeNone({self._content!r})"


class OneOf:
    def __init__(self, contents):
        self._contents = contents

    @property
    def contents(self):
        return self._contents

    def __eq__(self, other):
        if isinstance(other, OneOf):
            return set(self._contents) == set(other._contents)
        else:
            return False

    def __repr__(self):
        return f"OneOf({self._contents!r})"


def _length_after_slice(slice, original_length):
    start, stop, step = slice.indices(original_length)
    assert step != 0

    if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
        d, m = divmod(abs(start - stop), abs(step))
        return d + (1 if m != 0 else 0)
    else:
        return 0


class TypeTracerArray:
    @classmethod
    def from_array(cls, array, dtype=None):
        if isinstance(array, ak._v2.index.Index):
            array = array.data
        if dtype is None:
            dtype = array.dtype
        return cls(dtype, shape=array.shape)

    def __init__(self, dtype, shape=None):
        self._dtype = np.dtype(dtype)
        self.shape = shape

    def __repr__(self):
        dtype = repr(self._dtype)
        shape = ""
        if self._shape != (UnknownLength,):
            shape = ", " + repr(self._shape)
        return f"TypeTracerArray({dtype}{shape})"

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if ak._v2._util.isint(value):
            value = (value,)
        elif value is None or isinstance(value, (UnknownLengthType, UnknownScalar)):
            value = (UnknownLength,)
        elif not isinstance(value, tuple):
            value = tuple(value)
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

    def forget_length(self):
        return type(self)(self._dtype, (UnknownLength,) + self._shape[1:])

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

    class _CTypes:
        data = 0

    @property
    def ctypes(self):
        return self._CTypes

    def __len__(self):
        raise AssertionError(
            "bug in Awkward Array: attempt to get length of a TypeTracerArray"
        )

    def __setitem__(self, where, what):
        raise AssertionError(
            "bug in Awkward Array: attempt to set values of a TypeTracerArray"
        )

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

        if ak._v2._util.isint(where):
            if len(self._shape) == 1:
                if where == 0:
                    return UnknownScalar(self._dtype)
                else:
                    return UnknownScalar(self._dtype)
            else:
                return TypeTracerArray(self._dtype, self._shape[1:])

        elif isinstance(where, slice):
            return TypeTracerArray(self._dtype, (UnknownLength,) + self._shape[1:])

        elif (
            hasattr(where, "dtype")
            and hasattr(where, "shape")
            and issubclass(where.dtype.type, np.integer)
        ):
            assert len(self._shape) != 0
            return TypeTracerArray(self._dtype, where.shape + self._shape[1:])

        elif (
            hasattr(where, "dtype")
            and hasattr(where, "shape")
            and issubclass(where.dtype.type, (np.bool_, bool))
        ):
            assert len(self._shape) != 0
            return TypeTracerArray(self._dtype, (UnknownLength,) + self._shape[1:])

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
                if ak._v2._util.isint(wh):
                    shapes.append(numpy.array(0))
                elif hasattr(wh, "dtype") and hasattr(wh, "shape"):
                    sh = [
                        1 if isinstance(x, UnknownLengthType) else int(x)
                        for x in wh.shape
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
            assert len(shape) != 0

            return TypeTracerArray(self._dtype, (UnknownLength,) + shape[1:])

        elif (
            isinstance(where, tuple)
            and len(where) > 0
            and (ak._v2._util.isint(where[0]) or isinstance(where[0], slice))
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
            return TypeTracerArray(self._dtype, shape)

        else:
            raise NotImplementedError(repr(where))

    def __lt__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, numbers.Real):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        assert len(args) != 0
        assert ak._v2._util.isint(args[0]) or isinstance(args[0], UnknownLengthType)
        assert all(ak._v2._util.isint(x) for x in args[1:])
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

    def raw(self, array, nplike):
        assert isinstance(array.nplike, TypeTracer)

        if isinstance(nplike, TypeTracer):
            return TypeTracerArray.from_array(array)
        elif isinstance(array, TypeTracerArray):
            return self
        elif hasattr(nplike, "known_data") and nplike.known_data:
            raise TypeError(
                "Converting a TypeTracer nplike to a nplike with `known_data=True` is not possible"
            )
        else:
            raise TypeError(
                "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer"
            )

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
        return TypeTracerArray(dtype, shape)

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
        assert 1 <= len(args) <= 3
        assert (
            "dtype" in kwargs
        ), "internal error: calling arange without dtype (platform dependence)"

        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[0], args[1], 1
        elif len(args) == 3:
            start, stop, step = args[0], args[1], args[2]

        if (
            ak._v2._util.isint(start)
            and ak._v2._util.isint(stop)
            and ak._v2._util.isint(step)
        ):
            length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)

        return TypeTracerArray(kwargs["dtype"], (length,))

    def meshgrid(self, *args, **kwargs):
        # *arrays, indexing="ij"
        raise NotImplementedError

    ############################ testing

    def shape(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def array_equal(self, *args, **kwargs):
        # array1, array2
        return False

    def size(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def searchsorted(self, *args, **kwargs):
        # haystack, needle, side="right"
        raise NotImplementedError

    def argsort(self, array, *args, **kwargs):
        # array
        return TypeTracerArray(np.int64, array.shape)

    ############################ manipulation

    def broadcast_arrays(self, *arrays):
        # array1[, array2[, ...]]

        if len(arrays) == 0:
            return []

        next = []
        maxdim = 0
        for x in arrays:
            if not hasattr(x, "shape"):
                next.append(numpy.array(x))
            else:
                next.append(x)
                maxdim = max(maxdim, len(x.shape))

        if maxdim == 0:
            return next

        first, *rest = next
        shape = list(first.shape[1:])
        for x in rest:
            thisshape = x.shape[1:]
            if len(shape) < len(thisshape):
                shape = [1] * (len(thisshape) - len(shape)) + shape
            elif len(shape) > len(thisshape):
                thisshape = (1,) * (len(shape) - len(thisshape)) + thisshape
            for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
                if shape[i] == 1 and thisshape[i] != 1:
                    shape[i] = thisshape[i]
                elif shape[i] != 1 and thisshape[i] != 1 and shape[i] != thisshape[i]:
                    raise ValueError(
                        "shape mismatch: objects cannot be broadcast to a single shape"
                    )

        return [
            TypeTracerArray(x.dtype, [UnknownLength] + shape) for x in [first] + rest
        ]

    def add(self, *args, **kwargs):
        # array1, array2[, out=]
        raise NotImplementedError

    def cumsum(self, *args, **kwargs):
        # arrays[, out=]
        raise NotImplementedError

    def cumprod(self, *args, **kwargs):
        # arrays[, out=]
        raise NotImplementedError

    def nonzero(self, array):
        # array
        return (TypeTracerArray(np.int64, (UnknownLength,)),) * len(array.shape)

    def unique(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def concatenate(self, arrays):
        inner_shape = None
        emptyarrays = []
        for x in arrays:
            if inner_shape is None:
                inner_shape = x.shape[1:]
            elif inner_shape != x.shape[1:]:
                raise ValueError(
                    "inner dimensions don't match in concatenate: {} vs {}".format(
                        inner_shape, x.shape[1:]
                    )
                )
            emptyarrays.append(_emptyarray(x))

        if inner_shape is None:
            raise ValueError("need at least one array to concatenate")

        return TypeTracerArray(
            numpy.concatenate(emptyarrays).dtype, (UnknownLength,) + inner_shape
        )

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
