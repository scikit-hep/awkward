# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: keep this file, but modernize the 'of' function; ptr_lib is gone.

import ctypes

from collections.abc import Iterable

import numpy

import awkward as ak


class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class NumpyMetadata(Singleton):
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    longlong = numpy.longlong
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    str_ = numpy.str_
    bytes_ = numpy.bytes_

    intp = numpy.intp
    integer = numpy.integer
    signedinteger = numpy.signedinteger
    unsignedinteger = numpy.unsignedinteger
    floating = numpy.floating
    complexfloating = numpy.complexfloating
    number = numpy.number
    object_ = numpy.object_
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    errstate = numpy.errstate
    newaxis = numpy.newaxis

    ndarray = numpy.ndarray

    nan = numpy.nan
    inf = numpy.inf

    nat = numpy.datetime64("NaT")
    datetime_data = numpy.datetime_data
    issubdtype = numpy.issubdtype

    AxisError = numpy.AxisError


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
    known_data = True
    known_shape = True
    known_dtype = True

    ############################ array creation

    def array(self, *args, **kwargs):
        # data[, dtype=[, copy=]]
        return self._module.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        # array[, dtype=][, order=]
        return self._module.asarray(*args, **kwargs)

    def ascontiguousarray(self, *args, **kwargs):
        # array[, dtype=]
        return self._module.ascontiguousarray(*args, **kwargs)

    def isscalar(self, *args, **kwargs):
        return self._module.isscalar(*args, **kwargs)

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

    def zeros_like(self, *args, **kwargs):
        # array
        return self._module.zeros_like(*args, **kwargs)

    def ones_like(self, *args, **kwargs):
        # array
        return self._module.ones_like(*args, **kwargs)

    def full_like(self, *args, **kwargs):
        # array, fill_value
        return self._module.full_like(*args, **kwargs)

    def arange(self, *args, **kwargs):
        # stop[, dtype=]
        # start, stop[, dtype=]
        # start, stop, step[, dtype=]
        return self._module.arange(*args, **kwargs)

    def meshgrid(self, *args, **kwargs):
        # *arrays, indexing="ij"
        return self._module.meshgrid(*args, **kwargs)

    ############################ testing

    def shape(self, *args, **kwargs):
        # array
        return self._module.shape(*args, **kwargs)

    def array_equal(self, *args, **kwargs):
        # array1, array2
        return self._module.array_equal(*args, **kwargs)

    def size(self, *args, **kwargs):
        # array
        return self._module.size(*args, **kwargs)

    def searchsorted(self, *args, **kwargs):
        # haystack, needle, side="right"
        return self._module.searchsorted(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        # array
        return self._module.argsort(*args, **kwargs)

    ############################ manipulation

    def broadcast_arrays(self, *args, **kwargs):
        # array1[, array2[, ...]]
        return self._module.broadcast_arrays(*args, **kwargs)

    def cumsum(self, *args, **kwargs):
        # arrays[, out=]
        return self._module.cumsum(*args, **kwargs)

    def cumprod(self, *args, **kwargs):
        # arrays[, out=]
        return self._module.cumprod(*args, **kwargs)

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

    def tile(self, *args, **kwargs):
        # array, int
        return self._module.tile(*args, **kwargs)

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

    def append(self, *args, **kwargs):
        # array, element
        return self._module.append(*args, **kwargs)

    def where(self, *args, **kwargs):
        # array, element
        return self._module.where(*args, **kwargs)

    ############################ ufuncs

    def add(self, *args, **kwargs):
        # array1, array2
        return self._module.add(*args, **kwargs)

    def multiply(self, *args, **kwargs):
        # array1, array2
        return self._module.multiply(*args, **kwargs)

    def logical_or(self, *args, **kwargs):
        # array1, array2
        return self._module.logical_or(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        # array1, array2
        return self._module.logical_and(*args, **kwargs)

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

    def equal(self, *args, **kwargs):
        # array1, array2
        return self._module.equal(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        # array
        return self._module.ceil(*args, **kwargs)

    def minimum(self, *args, **kwargs):
        # array1, array2
        return self._module.minimum(*args, **kwargs)

    def maximum(self, *args, **kwargs):
        # array1, array2
        return self._module.maximum(*args, **kwargs)

    ############################ almost-ufuncs

    def nan_to_num(self, *args, **kwargs):
        # array, copy=True, nan=0.0, posinf=None, neginf=None
        return self._module.nan_to_num(*args, **kwargs)

    def isclose(self, *args, **kwargs):
        # a, b, rtol=1e-05, atol=1e-08, equal_nan=False
        return self._module.isclose(*args, **kwargs)

    def isnan(self, *args, **kwargs):
        # array
        return self._module.isnan(*args, **kwargs)

    def isneginf(self, *args, **kwargs):
        # array
        return self._module.isneginf(*args, **kwargs)

    def isposinf(self, *args, **kwargs):
        # array
        return self._module.isposinf(*args, **kwargs)

    def isfinite(self, *args, **kwargs):
        # array
        return self._module.isfinite(*args, **kwargs)

    ############################ reducers

    def all(self, *args, **kwargs):
        # array
        kwargs.pop("prefer", None)
        return self._module.all(*args, **kwargs)

    def any(self, *args, **kwargs):
        # array
        kwargs.pop("prefer", None)
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

    def array_str(self, *args, **kwargs):
        # array, max_line_width, precision=None, suppress_small=None
        return self._module.array_str(*args, **kwargs)

    def datetime_as_string(self, *args, **kwargs):
        return self._module.datetime_as_string(*args, **kwargs)


class NumpyKernel:
    def __init__(self, kernel, name_and_types):
        self._kernel = kernel
        self._name_and_types = name_and_types

    def __repr__(self):
        return "<{} {}{}>".format(
            type(self).__name__,
            self._name_and_types[0],
            "".join(", " + str(numpy.dtype(x)) for x in self._name_and_types[1:]),
        )

    @staticmethod
    def _cast(x, t):
        if issubclass(t, ctypes._Pointer):

            if is_numpy_buffer(x):
                return ctypes.cast(x.ctypes.data, t)
            elif is_cupy_buffer(x):
                raise ak._v2._util.error(
                    AssertionError("CuPy buffers shouldn't be passed to Numpy Kernels.")
                )
            elif is_jax_buffer(x):
                raise ak._v2._util.error(
                    ValueError(
                        "JAX Buffers can't be passed as function args for the C Kernels"
                    )
                )
            else:
                return ctypes.cast(x, t)
        else:
            return x

    def __call__(self, *args):
        assert len(args) == len(self._kernel.argtypes)

        if not any(is_jax_tracer(arg) for arg in args):
            return self._kernel(
                *(self._cast(x, t) for x, t in zip(args, self._kernel.argtypes))
            )


class CupyKernel(NumpyKernel):
    def max_length(self, args):
        cupy = ak._v2._connect.cuda.import_cupy("Awkward Arrays with CUDA")
        max_length = numpy.iinfo(numpy.int64).min
        for array in args:
            if isinstance(array, cupy.ndarray):
                max_length = max(max_length, len(array))
        return max_length

    def calc_grid(self, length):
        if length > 1024:
            return -(length // -1024), 1, 1
        return 1, 1, 1

    def calc_blocks(self, length):
        if length > 1024:
            return 1024, 1, 1
        return length, 1, 1

    def __call__(self, *args):
        cupy = ak._v2._connect.cuda.import_cupy("Awkward Arrays with CUDA")
        maxlength = self.max_length(args)
        grid, blocks = self.calc_grid(maxlength), self.calc_blocks(maxlength)
        cupy_stream_ptr = cupy.cuda.get_current_stream().ptr

        if cupy_stream_ptr not in ak._v2._connect.cuda.cuda_streamptr_to_contexts:
            ak._v2._connect.cuda.cuda_streamptr_to_contexts[cupy_stream_ptr] = (
                cupy.array(ak._v2._connect.cuda.NO_ERROR),
                [],
            )

        assert len(args) == len(self._kernel.dir)
        # The first arg is the invocation index which raises itself by 8 in the kernel if there was no error before.
        # The second arg is the error_code.
        args = list(args)
        args.extend(
            [
                len(
                    ak._v2._connect.cuda.cuda_streamptr_to_contexts[cupy_stream_ptr][1]
                ),
                ak._v2._connect.cuda.cuda_streamptr_to_contexts[cupy_stream_ptr][0],
            ]
        )
        ak._v2._connect.cuda.cuda_streamptr_to_contexts[cupy_stream_ptr][1].append(
            ak._v2._connect.cuda.Invocation(
                name=self._name_and_types[0],
                error_context=ak._v2._util.ErrorContext.primary(),
            )
        )

        self._kernel(grid, blocks, tuple(args))


class Numpy(NumpyLike):
    @property
    def index_nplike(self):
        return self

    def to_rectilinear(self, array, *args, **kwargs):
        if isinstance(array, numpy.ndarray):
            return array

        elif isinstance(
            array,
            (
                ak.Array,
                ak.Record,
                ak.ArrayBuilder,
                ak.layout.Content,
                ak.layout.Record,
                ak.layout.ArrayBuilder,
                ak.layout.LayoutBuilder32,
                ak.layout.LayoutBuilder64,
            ),
        ):
            return ak.operations.convert.to_numpy(array, *args, **kwargs)

        elif isinstance(array, Iterable):
            return [self.to_rectilinear(x, *args, **kwargs) for x in array]

        else:
            raise TypeError("to_rectilinear argument must be iterable")

    def __getitem__(self, name_and_types):
        return NumpyKernel(ak._cpu_kernels.kernel[name_and_types], name_and_types)

    def __init__(self):
        self._module = numpy

    @property
    def ma(self):
        return self._module.ma

    @property
    def char(self):
        return self._module.char

    @property
    def ndarray(self):
        return self._module.ndarray

    def raw(self, array, nplike):
        if isinstance(nplike, Numpy):
            return array
        elif isinstance(nplike, Cupy):
            cupy = Cupy.instance()
            return cupy.asarray(array, dtype=array.dtype, order="C")
        elif isinstance(nplike, ak._v2._typetracer.TypeTracer):
            return ak._v2._typetracer.TypeTracerArray(
                dtype=array.dtype, shape=array.shape
            )
        elif isinstance(nplike, Jax):
            jax = Jax.instance()
            return jax.asarray(array, dtype=array.dtype)
        else:
            raise TypeError(
                "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax"
            )


class Cupy(NumpyLike):
    @property
    def index_nplike(self):
        return self

    def to_rectilinear(self, array, *args, **kwargs):
        return ak.operations.convert.to_cupy(array, *args, **kwargs)

    def __getitem__(self, name_and_types):
        cupy = ak._v2._connect.cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = ak._v2._connect.cuda.initialize_cuda_kernels(cupy)  # noqa: F401

        func = _cuda_kernels[name_and_types]
        if func is not None:
            return CupyKernel(func, name_and_types)
        else:
            raise NotImplementedError(
                f"{name_and_types[0]} is not implemented for CUDA. Please transfer the array back to the Main Memory to "
                "continue the operation."
            )

    def __init__(self):
        import awkward._v2._connect.cuda  # noqa: F401

        self._module = ak._v2._connect.cuda.import_cupy("Awkward Arrays with CUDA")

    @property
    def ma(self):
        raise ValueError(
            "CUDA arrays cannot have missing values until CuPy implements "
            "numpy.ma.MaskedArray" + ak._util.exception_suffix(__file__)
        )

    @property
    def char(self):
        raise ValueError(
            "CUDA arrays cannot do string manipulations until CuPy implements "
            "numpy.char" + ak._util.exception_suffix(__file__)
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def asarray(self, array, dtype=None, order=None):
        if isinstance(
            array,
            (
                ak.highlevel.Array,
                ak.highlevel.Record,
                ak.layout.Content,
                ak.layout.Record,
            ),
        ):
            out = ak.operations.convert.to_cupy(array)
            if dtype is not None and out.dtype != dtype:
                return self._module.asarray(out, dtype=dtype, order=order)
            else:
                return out
        else:
            return self._module.asarray(array, dtype=dtype, order=order)

    def raw(self, array, nplike):
        if isinstance(nplike, Cupy):
            return array
        elif isinstance(nplike, Numpy):
            numpy = Numpy.instance()
            return numpy.asarray(array.get(), dtype=array.dtype, order="C")
        elif isinstance(nplike, ak._v2._typetracer.TypeTracer):
            return ak._v2._typetracer.TypeTracerArray(
                dtype=array.dtype, shape=array.shape
            )
        elif isinstance(nplike, Jax):
            jax = Jax.instance()
            return jax.asarray(array.get(), dtype=array.dtype)
        else:
            raise TypeError(
                "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax"
            )

    def ascontiguousarray(self, array, dtype=None):
        if isinstance(
            array,
            (
                ak.highlevel.Array,
                ak.highlevel.Record,
                ak.layout.Content,
                ak.layout.Record,
            ),
        ):
            out = ak.operations.convert.to_cupy(array)
            if dtype is not None and out.dtype != dtype:
                return self._module.ascontiguousarray(out, dtype=dtype)
            else:
                return out
        else:
            return self._module.ascontiguousarray(array, dtype=dtype)

    def zeros(self, *args, **kwargs):
        return self._module.zeros(*args, **kwargs)

    def frombuffer(self, *args, **kwargs):
        np_array = numpy.frombuffer(*args, **kwargs)
        return self._module.array(np_array)

    def array_equal(self, array1, array2):
        # CuPy issue?
        if array1.shape != array2.shape:
            return False
        else:
            return self._module.all(array1 - array2 == 0)

    def repeat(self, array, repeats):
        # https://github.com/cupy/cupy/issues/3849
        if isinstance(repeats, self._module.ndarray):
            all_stops = self._module.cumsum(repeats)
            parents = self._module.zeros(all_stops[-1].item(), dtype=int)
            stops, stop_counts = self._module.unique(all_stops[:-1], return_counts=True)
            parents[stops] = stop_counts
            self._module.cumsum(parents, out=parents)
            return array[parents]
        else:
            return self._module.repeat(array, repeats)

    def nan_to_num(self, *args, **kwargs):
        self._module.nan_to_num(*args, **kwargs)

    # For all reducers: https://github.com/cupy/cupy/issues/3819

    def all(self, array, axis=None, **kwargs):
        kwargs.pop("prefer", None)
        out = self._module.all(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def any(self, array, axis=None, **kwargs):
        kwargs.pop("prefer", None)
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

    def array_str(
        self, array, max_line_width=None, precision=None, suppress_small=None
    ):
        # array, max_line_width, precision=None, suppress_small=None
        return self._module.array_str(array, max_line_width, precision, suppress_small)


class Jax(NumpyLike):
    @property
    def index_nplike(self):
        return ak.nplike.Numpy.instance()

    def to_rectilinear(self, array, *args, **kwargs):
        if isinstance(array, self._module.DeviceArray):
            return array

        elif isinstance(
            array,
            (
                ak.Array,
                ak.Record,
                ak.ArrayBuilder,
                ak.layout.Content,
                ak.layout.Record,
                ak.layout.ArrayBuilder,
                ak.layout.LayoutBuilder32,
                ak.layout.LayoutBuilder64,
            ),
        ):
            return ak.operations.convert.to_jax(array, *args, **kwargs)

        elif isinstance(array, Iterable):
            return [self.to_rectilinear(x, *args, **kwargs) for x in array]

        else:
            raise ak._v2._util.error(
                ValueError("to_rectilinear argument must be iterable")
            )

    def __getitem__(self, name_and_types):
        return NumpyKernel(ak._cpu_kernels.kernel[name_and_types], name_and_types)

    def __init__(self):
        from awkward._v2._connect.jax import import_jax  # noqa: F401

        self._module = import_jax().numpy

    @property
    def ma(self):
        ak._v2._util.error(
            ValueError(
                "JAX arrays cannot have missing values until JAX implements "
                "numpy.ma.MaskedArray" + ak._util.exception_suffix(__file__)
            )
        )

    @property
    def char(self):
        ak._v2._util.error(
            ValueError(
                "JAX arrays cannot do string manipulations until JAX implements "
                "numpy.char"
            )
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def asarray(self, array, dtype=None, order=None):
        return self._module.asarray(array, dtype=dtype, order="K")

    def ascontiguousarray(self, array, dtype=None):
        if isinstance(
            array,
            (
                ak.highlevel.Array,
                ak.highlevel.Record,
                ak.layout.Content,
                ak.layout.Record,
            ),
        ):
            out = ak.operations.convert.to_jax(array)
            if dtype is not None and out.dtype != dtype:
                return self._module.ascontiguousarray(out, dtype=dtype)
            else:
                return out
        else:
            return self._module.ascontiguousarray(array, dtype=dtype)

    def raw(self, array, nplike):
        if isinstance(nplike, Jax):
            return array
        elif isinstance(nplike, ak.nplike.Cupy):
            cupy = ak.nplike.Cupy.instance()
            return cupy.asarray(array)
        elif isinstance(nplike, ak.nplike.Numpy):
            numpy = ak.nplike.Numpy.instance()
            return numpy.asarray(array)
        elif isinstance(nplike, ak._v2._typetracer.TypeTracer):
            return ak._v2._typetracer.TypeTracerArray(
                dtype=array.dtype, shape=array.shape
            )
        else:
            ak._v2._util.error(
                TypeError(
                    "Invalid nplike, choose between nplike.Numpy, nplike.Cupy, Typetracer or Jax",
                )
            )

    # For all reducers: JAX returns zero-dimensional arrays like CuPy

    def all(self, *args, **kwargs):
        out = self._module.all(*args, **kwargs)
        return out

    def any(self, *args, **kwargs):
        out = self._module.any(*args, **kwargs)
        return out

    def count_nonzero(self, *args, **kwargs):
        out = self._module.count_nonzero(*args, **kwargs)
        return out

    def sum(self, *args, **kwargs):
        out = self._module.sum(*args, **kwargs)
        return out

    def prod(self, *args, **kwargs):
        out = self._module.prod(*args, **kwargs)
        return out

    def min(self, *args, **kwargs):
        out = self._module.min(*args, **kwargs)
        return out

    def max(self, *args, **kwargs):
        out = self._module.max(*args, **kwargs)
        return out

    def argmin(self, *args, **kwargs):
        out = self._module.argmin(*args, **kwargs)
        return out

    def argmax(self, *args, **kwargs):
        out = self._module.argmax(*args, **kwargs)
        return out


def is_numpy_buffer(obj) -> bool:
    """
    Args:
        obj: object to test

    Return `True` if the given object is a numpy buffer, otherwise `False`.

    """
    return isinstance(obj, numpy.ndarray)


def is_cupy_buffer(obj) -> bool:
    """
    Args:
        obj: object to test

    Return `True` if the given object is a cupy buffer, otherwise `False`.

    """
    module, _, suffix = type(obj).__module__.partition(".")
    return module == "cupy"


def is_jax_buffer(obj) -> bool:
    """
    Args:
        obj: object to test

    Return `True` if the given object is a jax buffer, otherwise `False`.

    """
    module, _, suffix = type(obj).__module__.partition(".")
    return module == "jaxlib"


def is_jax_tracer(obj) -> bool:
    """
    Args:
        obj: object to test

    Return `True` if the given object is a jax tracer, otherwise `False`.

    """
    module, _, suffix = type(obj).__module__.partition(".")
    return module == "jax"


def of(*arrays, default_cls=Numpy):
    """
    Args:
        *arrays: iterable of possible array objects
        default_cls: default NumpyLike class if no array objects found

    Return the #ak.nplike.NumpyLike that is best-suited to operating upon the given
    iterable of arrays. Return an instance of the `default_cls` if no known array types
    are found.
    """
    nplikes = set()
    for array in arrays:
        nplike = getattr(array, "nplike", None)
        if nplike is not None:
            nplikes.add(nplike)
        elif is_numpy_buffer(array):
            nplikes.add(Numpy.instance())
        elif is_cupy_buffer(array):
            nplikes.add(Cupy.instance())
        elif is_jax_buffer(array):
            nplikes.add(Jax.instance())

    if any(isinstance(x, ak._v2._typetracer.TypeTracer) for x in nplikes):
        return ak._v2._typetracer.TypeTracer.instance()

    if nplikes == set():
        return default_cls.instance()
    elif len(nplikes) == 1:
        return next(iter(nplikes))
    else:
        raise ValueError(
            """attempting to use both a 'cpu' array and a 'cuda' array in the """
            """same operation; use one of

    ak.to_backend(array, 'cpu')
    ak.to_backend(array, 'cuda')

to move one or the other to main memory or the GPU(s)."""
            + ak._util.exception_suffix(__file__)
        )
