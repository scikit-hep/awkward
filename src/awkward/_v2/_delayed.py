# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys
import threading
import queue

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Future:
    def __init__(self, task):
        # called by the main thread
        self._task = task
        self._finished = threading.Event()
        self._result = None
        self._exc_info = None
        self._error_context = ak._v2._util.ErrorContext.primary()

    @property
    def is_exception(self):
        return self.exc_info is not None

    def __repr__(self):
        return f"Future({self._task})"

    def run(self):
        # on the Worker thread
        try:
            self._result = self._task()
        except Exception:
            self._exc_info = sys.exc_info()
        finally:
            self._finished.set()

    def result(self):
        # called by the main thread
        self._finished.wait()
        if self.is_exception:
            exception_class, exception_value, traceback = self._exc_info
            raise ak._v2._util.error(
                exception_value.with_traceback(traceback),
                error_context=self._error_context,
            )
        else:
            return self._result


class Worker(threading.Thread):
    def __init__(self):
        # called by the main thread
        super().__init__(daemon=True)
        self._tasks = queue.Queue()

    def run(self):
        # on the Worker thread
        while True:
            task = self._tasks.get()
            if not isinstance(task, Future):
                break
            task.run()
            if task.is_exception:
                break


class DelayedArray:
    def __init__(self, shape, dtype, future):
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._future = future

    def __repr__(self):
        return f"DelayedArray({self._shape}, {self._dtype}, {self._future})"

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        else:
            return self._future.result().shape

    @property
    def strides(self):
        return self._future.result().strides

    @property
    def nplike(self):
        return Delayed.instance()

    @property
    def ndim(self):
        return len(self.shape)

    def __iter__(self):
        raise NotImplementedError

    def __array__(self, *args, **kwargs):
        raise NotImplementedError

    def itemsize(self):
        return self._dtype.itemsize

    class _CTypes:
        pass

    @property
    def ctypes(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

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

    next_nplike = ak.nplike.Numpy.instance()

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
