# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import threading
import typing as tp
from contextlib import contextmanager

from awkward._nplikes.array_like import ArrayLike

if tp.TYPE_CHECKING:
    from awkward._kernels import Kernel
    from awkward._nplikes.numpy_like import NumpyLike
    from awkward._nplikes.typetracer import TypeTracerArray


# make this somehow a class level attribute to further hide it in a namespace?
# it's off by default, you need to use `trace` to make it work
_enable_adding_new_main_trace: bool = False


@contextmanager
def disable_adding_new_main_trace():
    global _enable_adding_new_main_trace
    oldval = _enable_adding_new_main_trace
    _enable_adding_new_main_trace = False
    try:
        yield
    finally:
        _enable_adding_new_main_trace = oldval


class ArbitraryFunction:
    __slots__ = ("func",)

    def __init__(self, func: tp.Callable):
        self.func = func

    def get(self, *args, **kwargs):
        return self.func


class AwkwardKernelFunction:
    def __init__(self, instance: Kernel, func: tp.Callable):
        self.instance = instance
        self.func = func

    def get(self):
        return getattr(self.instance, self.func.__name__)


class NumpyLikeFunction:
    def __init__(self, instance: NumpyLike, func: tp.Callable):
        self.instance = instance
        self.func = func

    def get(self):
        return getattr(self.instance, self.func.__name__)


TraceableFunctions = tp.Union[
    ArbitraryFunction, AwkwardKernelFunction, NumpyLikeFunction
]


class FuncCapture(tp.NamedTuple):
    # a computation applied to one or more buffers
    func: TraceableFunctions
    args: tuple[tp.Any, ...] = ()
    kwargs: dict[str, tp.Any] = {}

    def __call__(self, *in_arrays: ArrayLike) -> ArrayLike | tuple[ArrayLike, ...]:
        return self.func.get()(*in_arrays, *self.args, **self.kwargs)


class MainTrace(tp.NamedTuple):
    # a trace of a single computation
    level: int
    func_capture: FuncCapture
    in_arrays: tuple[TypeTracerArray, ...]
    out_arrays: tuple[TypeTracerArray, ...]


class TraceStack(threading.local):
    __slots__ = ("stack",)

    def __init__(self) -> None:
        self.stack: list[MainTrace] = []

    def push(self, trace: MainTrace) -> None:
        self.stack.append(trace)

    def pop(self) -> MainTrace:
        return self.stack.pop()

    def __len__(self) -> int:
        return len(self.stack)

    def __iter__(self) -> tp.Iterator[MainTrace]:
        return iter(self.stack)

    def __getitem__(self, key) -> MainTrace:
        return self.stack[key]

    def copy(self) -> TraceStack:
        new_stack = TraceStack()
        new_stack.stack = self.stack.copy()
        return new_stack


def try_merging_last_two_traces(
    second_to_last_trace: MainTrace, last_trace: MainTrace
) -> tuple[MainTrace, ...]:
    # Merge the last two traces if possible,
    # this could reduce the number of traces
    # in the stack and may improve performance
    # as we loop over the stack to compute the
    # final result.

    # this means we can merge them
    # TODO(pfackeldey): compare tracers based on their properties (dtype, shape, form_key)?
    # the current implementation is not correct
    if False and second_to_last_trace.out_arrays == last_trace.in_arrays:
        new_last_trace = MainTrace(
            level=second_to_last_trace.level,
            func_capture=second_to_last_trace.func_capture.merge(
                last_trace.func_capture
            ),
            in_arrays=second_to_last_trace.in_arrays,
            out_arrays=last_trace.out_arrays,
        )
        return (new_last_trace,)
    return (second_to_last_trace, last_trace)


@contextmanager
def new_main_trace(
    func_capture: FuncCapture,
    in_arrays: TypeTracerArray | tuple[TypeTracerArray, ...],
    out_arrays: TypeTracerArray | tuple[TypeTracerArray, ...],
    optimize: bool = False,
):
    if _enable_adding_new_main_trace:
        # make sure we have tuples
        if not isinstance(in_arrays, tuple):
            in_arrays = (in_arrays,)
        if not isinstance(out_arrays, tuple):
            out_arrays = (out_arrays,)

        # create the new trace
        main_trace = MainTrace(
            level=len(trace_stack),
            func_capture=func_capture,
            in_arrays=in_arrays,
            out_arrays=out_arrays,
        )

        if main_trace.level > 0:
            # pop the previous trace from the stack
            prev_trace = trace_stack.pop()
            traces = (prev_trace, main_trace)

            # try to optimize on-the-fly
            if optimize and len(trace_stack) > 0:
                traces = try_merging_last_two_traces(*traces)

        else:
            # no previous trace, just push the new trace
            traces = (main_trace,)

        # push the new trace(s) to the stack
        for trace in traces:
            trace_stack.push(trace)

    yield


def trace(fun, *args, **kwargs) -> tuple[tp.Any, TraceStack]:
    # we need to make sure that we can add a new main trace.
    # this will only do something if used with the typetracer backend.
    global _enable_adding_new_main_trace
    _enable_adding_new_main_trace = True

    global trace_stack
    trace_stack = TraceStack()

    try:
        res = fun(*args, **kwargs)
        res_trace_stack = trace_stack.copy()
    finally:
        _enable_adding_new_main_trace = False
        # we return a copy and delete the original
        # to make sure we don't have any side effects
        # when we call trace again.
        del trace_stack
    return (res, res_trace_stack)
