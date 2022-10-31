# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers
from functools import reduce

import numpy

import awkward as ak
from awkward import index, nplikes

np = nplikes.NumpyMetadata.instance()


class NoError:
    def __init__(self):
        self.str = None
        self.filename = None
        self.pass_through = False
        self.attempt = ak._util.kSliceNone
        self.id = ak._util.kSliceNone


class NoKernel:
    def __call__(self, *args):
        return NoError()


class UnknownLengthType:

    @property
    def nplike(self):
        return TypeTracer.instance()

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

    @property
    def nplike(self):
        return TypeTracer.instance()

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
        # Always simplify `MaybeNone`s
        if isinstance(content, self.__class__):
            content = content.content
        self._content = content

    @property
    def nplike(self):
        return TypeTracer.instance()

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

    def __str__(self):
        return f"maybe-{self._content}"


class OneOf:
    def __init__(self, contents):
        self._contents = contents

    @property
    def contents(self):
        return self._contents

    @property
    def nplike(self):
        return TypeTracer.instance()

    def __eq__(self, other):
        if isinstance(other, OneOf):
            return set(self._contents) == set(other._contents)
        else:
            return False

    def __repr__(self):
        return f"OneOf({self._contents!r})"

    def __str__(self):
        return (
            f"oneof-{'-'.join(str(x).replace('unknown-', '') for x in self._contents)}"
        )


def _length_after_slice(slice, original_length):
    start, stop, step = slice.indices(original_length)
    assert step != 0

    if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
        d, m = divmod(abs(start - stop), abs(step))
        return d + (1 if m != 0 else 0)
    else:
        return 0


class TypeTracerArray:
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
        if ak._util.isint(value):
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

    def astype(self, dtype):
        return self.__class__(np.dtype(dtype), self._shape)

    def view(self, dtype):
        if (
            self.itemsize != np.dtype(dtype).itemsize
            and self._shape[-1] != UnknownLength
        ):
            last = int(
                round(self._shape[-1] * self.itemsize / np.dtype(dtype).itemsize)
            )
            shape = self._shape[:-1] + (last,)
        else:
            shape = self._shape
        dtype = np.dtype(dtype)
        return self.__class__(dtype, shape)

    def forget_length(self):
        return type(self)(self._dtype, (UnknownLength,) + self._shape[1:])

    def __iter__(self):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
            )
        )

    def __array__(self, *args, **kwargs):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to convert TypeTracerArray into a concrete array"
            )
        )

    @property
    def itemsize(self):
        return self._dtype.itemsize

    class _CTypes:
        data = 0

    @property
    def ctypes(self):
        return self._CTypes

    def __len__(self):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to get length of a TypeTracerArray"
            )
        )

    def __setitem__(self, where, what):
        raise ak._errors.wrap_error(
            AssertionError(
                "bug in Awkward Array: attempt to set values of a TypeTracerArray"
            )
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

        if ak._util.isint(where):
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
                if ak._util.isint(wh):
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
                    raise ak._errors.wrap_error(NotImplementedError(repr(wh)))

            slicer_shape = numpy.broadcast_arrays(*shapes)[0].shape

            shape = basic_shape + slicer_shape + self._shape[num_basic + len(shapes) :]
            assert len(shape) != 0

            return TypeTracerArray(self._dtype, (UnknownLength,) + shape[1:])

        elif (
            isinstance(where, tuple)
            and len(where) > 0
            and (ak._util.isint(where[0]) or isinstance(where[0], slice))
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
                    raise ak._errors.wrap_error(NotImplementedError(repr(wh)))

            shape = (next._shape[0],) + tuple(after_shape)
            return TypeTracerArray(self._dtype, shape)

        else:
            raise ak._errors.wrap_error(NotImplementedError(repr(where)))

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

    def __eq__(self, other):
        if ak._util.isscalar(other):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __ne__(self, other):
        if ak._util.isscalar(other):
            return TypeTracerArray(np.bool_, self._shape)
        else:
            return NotImplemented

    def __invert__(self):
        return TypeTracerArray(np.bool_, self._shape)

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        assert len(args) != 0
        assert ak._util.isint(args[0]) or isinstance(args[0], UnknownLengthType)
        assert all(ak._util.isint(x) for x in args[1:])
        assert all(x >= 0 for x in args[1:])

        return TypeTracerArray(self._dtype, (UnknownLength,) + args[1:])

    def copy(self):
        return self


class TypeTracer(ak.nplikes.NumpyLike):
    known_data = False
    known_shape = False

    @property
    def _module(self):
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def index_nplike(self):
        return self

    def to_rectilinear(self, array, *args, **kwargs):
        raise ak._errors.wrap_error(NotImplementedError)

    def __getitem__(self, name_and_types):
        return NoKernel()

    @property
    def ma(self):
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def char(self):
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def ndarray(self):
        return TypeTracerArray

    def raw(self, array, nplike):
        assert isinstance(array.nplike, TypeTracer)
        if nplike is not self:
            raise ak._errors.wrap_error(
                TypeError(
                    "Converting a TypeTracer nplike to a nplike with `known_data=True` is not possible"
                )
            )

        return array

    ############################ array creation

    def array(self, data, dtype=None, **kwargs):
        if isinstance(data, UnknownScalar):
            array_shape = ()
            array_dtype = data.dtype
        elif numpy.isscalar(data):
            data = numpy.array(data)
            array_shape = ()
            array_dtype = data.dtype
        elif isinstance(data, index.Index):
            array_shape = (data.length,)
            array_dtype = data.dtype
        elif hasattr(data, "shape"):
            array_shape = data.shape
            array_dtype = data.dtype
        elif ak._util.is_sized_iterable(data):
            dtypes = []

            # First, find dtype of non-UnknownScalar
            non_scalars = [x for x in data if not isinstance(x, UnknownScalar)]
            if len(non_scalars):
                non_scalar_array = numpy.array(non_scalars)
                dtypes.append(non_scalar_array.dtype)

            # Now merge in any UnknownScalar
            scalar_dtypes = [x.dtype for x in data if isinstance(x, UnknownScalar)]
            dtypes.extend(scalar_dtypes)

            if not dtypes:
                raise ak._errors.wrap_error(
                    ValueError(
                        f"bug in Awkward Array: attempt to construct `TypeTracerArray` "
                        f"from a sized iterable with no discernible dtype: {data}"
                    )
                )

            # Construct array
            array_dtype = reduce(numpy.promote_types, dtypes)
            array_shape = (len(data),)
        else:
            raise ak._errors.wrap_error(
                ValueError(
                    f"bug in Awkward Array: attempt to construct `TypeTracerArray` "
                    f"from a non-sized iterable: {data}"
                )
            )

        if array_dtype == np.dtype("O"):
            raise ak._errors.wrap_error(
                ValueError(
                    f"bug in Awkward Array: attempt to construct `TypeTracerArray` "
                    f"from a sequence of non-primitive types: {data}"
                )
            )

        if dtype is None:
            dtype = array_dtype

        return TypeTracerArray(dtype, shape=array_shape)

    def asarray(self, array, dtype=None, **kwargs):
        # array[, dtype=][, order=]
        return self.array(array, dtype=dtype)

    def ascontiguousarray(self, array, dtype=None, **kwargs):
        # array[, dtype=]
        return self.array(array, dtype=dtype)

    def isscalar(self, *args, **kwargs):
        raise ak._errors.wrap_error(NotImplementedError)

    def frombuffer(self, *args, **kwargs):
        # array[, dtype=]
        raise ak._errors.wrap_error(NotImplementedError)

    def zeros(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def ones(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def empty(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        return TypeTracerArray(dtype, shape)

    def full(self, shape, value, dtype=None, **kwargs):
        value_array = self.array(value, dtype=dtype)
        return TypeTracerArray(value_array.dtype, shape)

    def zeros_like(self, a, dtype=None, **kwargs):
        return self.array(a, dtype=dtype)

    def ones_like(self, a, dtype=None, **kwargs):
        return self.array(a, dtype=dtype)

    def full_like(self, a, fill_value, dtype=None, **kwargs):
        return self.array(a, dtype=dtype)

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
        else:
            raise ak._errors.wrap_error(ValueError("incorrect number of arguments"))

        if ak._util.isint(start) and ak._util.isint(stop) and ak._util.isint(step):
            length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
        else:
            length = UnknownLength
        return TypeTracerArray(kwargs["dtype"], (length,))

    def meshgrid(self, *args, **kwargs):
        # *arrays, indexing="ij"
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ testing

    def shape(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def array_equal(self, *args, **kwargs):
        # array1, array2
        return False

    def size(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def searchsorted(self, *args, **kwargs):
        # haystack, needle, side="right"
        raise ak._errors.wrap_error(NotImplementedError)

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
                    raise ak._errors.wrap_error(
                        ValueError(
                            "shape mismatch: objects cannot be broadcast to a single shape"
                        )
                    )

        return [
            TypeTracerArray(x.dtype, [UnknownLength] + shape) for x in [first] + rest
        ]

    def add(self, x, y):
        # array1, array2[, out=]
        is_array = False
        if isinstance(x, TypeTracerArray):
            is_array = True
            x = x[0]
        if isinstance(y, TypeTracerArray):
            is_array = True
            y = y[0]
        out = x + y
        if is_array:
            return TypeTracerArray(out.dtype)
        else:
            return out

    def multiply(self, x, y):
        # array1, array2[, out=]
        return self.add(x, y)

    def maximum(self, x, y):
        # array1, array2[, out=]
        is_array = False
        if isinstance(x, TypeTracerArray):
            is_array = True
            x = x[0]
        if isinstance(y, TypeTracerArray):
            is_array = True
            y = y[0]
        is_maybenone = False
        if isinstance(x, MaybeNone):
            is_maybenone = True
            x = x.content
        if isinstance(y, MaybeNone):
            is_maybenone = True
            y = y.content
        out = x + y
        if is_array:
            return TypeTracerArray(out.dtype)
        elif is_maybenone:
            return MaybeNone(out)
        else:
            return out

    def minimum(self, x, y):
        return self.maximum(x, y)

    def cumsum(self, *args, **kwargs):
        # arrays[, out=]
        raise ak._errors.wrap_error(NotImplementedError)

    def cumprod(self, *args, **kwargs):
        # arrays[, out=]
        raise ak._errors.wrap_error(NotImplementedError)

    def nonzero(self, array):
        # array
        return (TypeTracerArray(np.int64, (UnknownLength,)),) * len(array.shape)

    def unique(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def concatenate(self, arrays):
        inner_shape = None
        emptyarrays = []
        for x in arrays:
            if inner_shape is None:
                inner_shape = x.shape[1:]
            elif inner_shape != x.shape[1:]:
                raise ak._errors.wrap_error(
                    ValueError(
                        "inner dimensions don't match in concatenate: {} vs {}".format(
                            inner_shape, x.shape[1:]
                        )
                    )
                )
            emptyarrays.append(_emptyarray(x))

        if inner_shape is None:
            raise ak._errors.wrap_error(
                ValueError("need at least one array to concatenate")
            )

        return TypeTracerArray(
            numpy.concatenate(emptyarrays).dtype, (UnknownLength,) + inner_shape
        )

    def repeat(self, *args, **kwargs):
        # array, int
        # array1, array2
        raise ak._errors.wrap_error(NotImplementedError)

    def tile(self, *args, **kwargs):
        # array, int
        raise ak._errors.wrap_error(NotImplementedError)

    def stack(self, *args, **kwargs):
        # arrays
        raise ak._errors.wrap_error(NotImplementedError)

    def vstack(self, *args, **kwargs):
        # arrays
        raise ak._errors.wrap_error(NotImplementedError)

    def packbits(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def unpackbits(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def atleast_1d(self, *args, **kwargs):
        # *arrays
        raise ak._errors.wrap_error(NotImplementedError)

    def broadcast_to(self, *args, **kwargs):
        # array, shape
        raise ak._errors.wrap_error(NotImplementedError)

    def append(self, *args, **kwargs):
        # array, element
        raise ak._errors.wrap_error(NotImplementedError)

    def where(self, *args, **kwargs):
        # array, element
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ ufuncs

    def sqrt(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def exp(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def true_divide(self, x, y):
        # array1, array2
        is_array = False
        if isinstance(x, TypeTracerArray):
            is_array = True
        if isinstance(y, TypeTracerArray):
            is_array = True
        array_dtypes = []
        scalar_dtypes = []

        raise ak._errors.wrap_error(NotImplementedError)

    def bitwise_or(self, *args, **kwargs):
        # array1, array2[, out=output]
        raise ak._errors.wrap_error(NotImplementedError)

    def logical_and(self, x, y):
        # array1, array2
        is_array = False
        if isinstance(x, TypeTracerArray):
            is_array = True
        if isinstance(y, TypeTracerArray):
            is_array = True
        if is_array:
            return TypeTracerArray(np.dtype(np.bool_))
        else:
            return UnknownScalar(np.dtype(np.bool_))

    def logical_or(self, x, y):
        # array1, array2[, out=]
        is_array = False
        if isinstance(x, TypeTracerArray):
            is_array = True
        if isinstance(y, TypeTracerArray):
            is_array = True
        if is_array:
            return TypeTracerArray(np.dtype(np.bool_))
        else:
            return UnknownScalar(np.dtype(np.bool_))

    def logical_not(self, array):
        # array1[, out=]
        is_array = False
        if isinstance(array, TypeTracerArray):
            is_array = True
        if is_array:
            return TypeTracerArray(np.dtype(np.bool_))
        else:
            return UnknownScalar(np.dtype(np.bool_))

    def equal(self, *args, **kwargs):
        # array1, array2
        raise ak._errors.wrap_error(NotImplementedError)

    def ceil(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ almost-ufuncs

    def nan_to_num(self, *args, **kwargs):
        # array, copy=True, nan=0.0, posinf=None, neginf=None
        raise ak._errors.wrap_error(NotImplementedError)

    def isclose(self, *args, **kwargs):
        # a, b, rtol=1e-05, atol=1e-08, equal_nan=False
        raise ak._errors.wrap_error(NotImplementedError)

    ############################ reducers

    def all(self, array, prefer):
        # array
        return prefer

    def any(self, array, prefer):
        # array
        return prefer

    def count_nonzero(self, *args, **kwargs):
        # array
        raise ak._errors.wrap_error(NotImplementedError)

    def _reduce(self, array, axis=None, dtype=None, keepdims=False):
        if dtype is None:
            dtype = array.dtype

        assert dtype != np.dtype("O")

        if axis is None:
            if keepdims:
                return TypeTracerArray(dtype, shape=tuple([1] * array.ndim))
            else:
                return UnknownScalar(dtype)

        else:
            if axis < 0:
                axis = array.ndim + axis

            new_shape = list(array.shape)
            if keepdims:
                new_shape[axis] = 1
            else:
                del new_shape[axis]

            new_shape = tuple(new_shape)
            return TypeTracerArray(dtype, shape=new_shape)

    def sum(self, array, axis=None, dtype=None, out=None, keepdims=False):
        # array
        return self._reduce(array, axis, dtype, keepdims)

    def prod(self, array, axis=None, dtype=None, out=None, keepdims=False):
        # array
        return self._reduce(array, axis, dtype, keepdims)

    def min(self, array, axis=None, dtype=None, out=None, keepdims=False):
        # array
        return self._reduce(array, axis, dtype, keepdims)

    def max(self, array, axis=None, dtype=None, out=None, keepdims=False):
        # array
        return self._reduce(array, axis, dtype, keepdims)

    def argmin(self, array, axis=None, dtype=None, out=None, keepdims=False):
        # array
        return self._reduce(array, axis, np.int64, keepdims)

    def argmax(self, array, axis=None, dtype=None, out=None, keepdims=False):
        # array
        return self._reduce(array, axis, np.int64, keepdims)

    def array_str(
        self, array, max_line_width=None, precision=None, suppress_small=None
    ):
        # array, max_line_width, precision=None, suppress_small=None
        return "[?? ... ??]"

    def datetime_as_string(self, *args, **kwargs):
        raise ak._errors.wrap_error(NotImplementedError)

    def promote_types(self, type1, type2):
        return numpy.promote_types(type1, type2)

    @classmethod
    def is_own_array(cls, obj) -> bool:
        return isinstance(obj, TypeTracerArray)
