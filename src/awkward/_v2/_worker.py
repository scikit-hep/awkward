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
        # on the Worker thread
        ak._v2._util.ErrorContext.override(self._error_context)
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


class DeadQueue:
    def __init__(self, exc_info):
        self._exc_info = exc_info

    def put(self, future):
        exception_class, exception_value, traceback = self._exc_info
        raise exception_value.with_traceback(traceback)


class Worker(threading.Thread):
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
