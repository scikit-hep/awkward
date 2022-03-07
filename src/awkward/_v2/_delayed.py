# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

from awkward._v2._worker import Future
from functools import partial

np = ak.nplike.NumpyMetadata.instance()


class DelayedArray:
    def __init__(self, shape, ndim, is_contiguous, dtype, nplike, future):
        self._shape = shape
        self._ndim = ndim
        self._is_contiguous = is_contiguous
        self._dtype = np.dtype(dtype)
        self._nplike = nplike
        self._future = future

    def __repr__(self):
        return f"DelayedArray({self._shape}, {self._ndim}, {self._is_contiguous}, {self._dtype}, {self._future})"

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        if isinstance(self._shape, Future):
            return self._shape.result()
        else:
            return self._shape

    @property
    def strides(self):
        if self._is_contiguous:
            return (self._dtype.itemsize,)
        else:
            return self._future.result().strides

    @property
    def nplike(self):
        return self._nplike

    @property
    def ndim(self):
        return self._ndim

    def __iter__(self):
        yield from self.nplike.nested_nplike.asarray(self._future.result())

    def __array__(self, *args, **kwargs):
        return self.nplike.nested_nplike.asarray(self._future.result())

    def itemsize(self):
        return self._dtype.itemsize

    class _CTypes:
        pass

    @property
    def ctypes(self):
        raise NotImplementedError

    def __len__(self):
        return self.shape[0]

    def __setitem__(self, where, what):
        raise NotImplementedError

    def __getitem__(self, where):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def reshape(self, *args):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


unset = object()


class Delayed(ak.nplike.NumpyLike):
    known_data = True  # eventually known
    known_shape = True  # eventually known

    nested_nplike = None

    def to_rectilinear(self, array, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, name_and_types):
        raise NotImplementedError

    @property
    def ma(self):
        raise NotImplementedError

    @property
    def char(self):
        raise NotImplementedError

    @property
    def ndarray(self):
        raise NotImplementedError

    def raw(self, array, nplike):
        raise NotImplementedError

    ############################ array creation

    def array(self, data, dtype=unset, **kwargs):
        # data[, dtype=[, copy=]]
        raise NotImplementedError

    def asarray(self, array, dtype=unset, **kwargs):
        # array[, dtype=][, order=]
        raise NotImplementedError

    def ascontiguousarray(self, array, dtype=unset, **kwargs):
        # array[, dtype=]
        raise NotImplementedError

    def isscalar(self, *args, **kwargs):
        raise NotImplementedError

    def frombuffer(self, *args, **kwargs):
        # array[, dtype=]
        raise NotImplementedError

    def zeros(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        raise NotImplementedError

    def ones(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        raise NotImplementedError

    def empty(self, shape, dtype=np.float64, **kwargs):
        # shape/len[, dtype=]
        raise NotImplementedError

    def full(self, shape, value, dtype=unset, **kwargs):
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

    def argsort(self, array, *args, **kwargs):
        # array
        raise NotImplementedError

    ############################ manipulation

    def broadcast_arrays(self, *arrays):
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

    def nonzero(self, array):
        # array
        raise NotImplementedError

    def unique(self, *args, **kwargs):
        # array
        raise NotImplementedError

    def concatenate(self, arrays):
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
        raise NotImplementedError

    def datetime_as_string(self, *args, **kwargs):
        raise NotImplementedError


class NumpyDelayed(Delayed):
    nested_nplike = ak.nplike.Numpy.instance()


class CupyDelayed(Delayed):
    nested_nplike = ak.nplike.Cupy.instance()
    cupy = ak._v2._connect.cuda.import_cupy("Awkward CUDA")

    def asarray(self, array, dtype=None, order=None):
        if not dtype:
            dtype = array.dtype
        future = ak._v2._connect.cuda.cuda_worker_threads[
            self.cupy.cuda.get_current_stream().ptr
        ].schedule(
            partial(self.nested_nplike.asarray, array=array, dtype=dtype, order=order)
        )
        return ak._v2._delayed.DelayedArray(
            shape=array.shape,
            ndim=array.ndim,
            is_contiguous=None,
            dtype=dtype,
            nplike=CupyDelayed.instance(),
            future=future,
        )
