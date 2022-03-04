# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys
import threading
import queue

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Future:
    def __init__(self, task, worker):
        # called by the main thread
        self._task = task
        self._worker = worker
        self._finished = threading.Event()
        self._result = None
        self._exc_info = None
        self._error_context = ak._v2._util.ErrorContext.primary()

    @property
    def task(self):
        return self._task

    @property
    def worker(self):
        return self._worker

    @property
    def is_exception(self):
        return self._exc_info is not None

    @property
    def exc_info(self):
        return self._exc_info

    def __repr__(self):
        return f"Future({self._task}, {self._worker})"

    def run(self):
<<<<<<< HEAD
        # on the Shadow thread
        # TO DO: set the ErrorContext to self._error_context
=======
        # on the Worker thread
        ak._v2._util.ErrorContext.override(self._error_context)
>>>>>>> 22b8184da46bc50b35a8b31dc4166c16bad76cf2
        try:
            self._result = self._task()
        except Exception:
            self._exc_info = sys.exc_info()
        finally:
            self._finished.set()

    def giveup(self, exc_info):
        # on the Worker thread
        self._exc_info = exc_info
        self._finished.set()

    def result(self):
        # called by the main thread
        self._finished.wait()

        if self.is_exception:
            exception_class, exception_value, traceback = self._exc_info
            raise exception_value.with_traceback(traceback)
        else:
            return self._result


<<<<<<< HEAD
# Hide in plain sight
class Shadow(threading.Thread):
=======
class DeadQueue:
    def __init__(self, exc_info):
        self._exc_info = exc_info

    def put(self, future):
        exception_class, exception_value, traceback = self._exc_info
        raise exception_value.with_traceback(traceback)


class Worker(threading.Thread):
>>>>>>> 22b8184da46bc50b35a8b31dc4166c16bad76cf2
    def __init__(self):
        # called by the main thread
        super().__init__(daemon=True)
        self._futures = getattr(queue, "SimpleQueue", queue.Queue)()

    def run(self):
        # on the Worker thread
        while True:
            future = self._futures.get()
            future.run()
            if future.is_exception:
                remaining = self._futures
                exc_info = future.exc_info
                # worker.schedule() will raise that exception henceforth
                self._futures = DeadQueue(exc_info)
                break

        # future.result() will raise that exception for all futures after the one that failed
        while not remaining.empty():
            future = remaining.get()
            future.giveup(exc_info)

    def schedule(self, task):
        # called by the main thread
        future = Future(task, self)
        self._futures.put(future)
        return future


class DelayedArray:
    def __init__(self, shape, ndim, is_contiguous, dtype, future):
        self._shape = shape
        self._ndim = ndim
        self._is_contiguous = is_contiguous
        self._dtype = np.dtype(dtype)
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
        return Delayed.instance()

    @property
    def ndim(self):
        return self._ndim

    def __iter__(self):
        yield from self.nplike.next_nplike.asarray(self._future.result())

    def __array__(self, *args, **kwargs):
        return self.nplike.next_nplike.asarray(self._future.result())

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

    next_nplike = None

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
    next_nplike = ak.nplike.Numpy.instance()

class CupyDelayed(Delayed):
    next_nplike = ak.nplike.Cupy.instance() 
