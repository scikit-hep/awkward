# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import typing as tp
from functools import wraps

from awkward.tracing.core import (
    ArbitraryFunction,
    AwkwardKernelFunction,
    FuncCapture,
    NumpyLikeFunction,
    _enable_adding_new_main_trace,
    new_main_trace,
)


def trace_typetracer_method(fun: tp.Callable) -> tp.Callable:
    """
    Decorator to trace the function calls
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        from awkward._kernels import TypeTracerKernel
        from awkward._nplikes.typetracer import TypeTracer, TypeTracerArray

        # return the result directly if we are not
        # allowed to add a new main trace for this one
        if not _enable_adding_new_main_trace:
            return fun(*args, **kwargs)

        del args

        # first construct the function
        inst = kwargs.pop("self", None)
        if inst is None:
            func = ArbitraryFunction(func=fun)
        elif isinstance(inst, TypeTracerKernel):
            func = AwkwardKernelFunction(instance=inst, func=fun)
        else:
            assert isinstance(inst, TypeTracer), (
                f"Expected TypeTracer, got {type(inst)}"
            )
            func = NumpyLikeFunction(instance=inst, func=fun)

        tracers = []
        remaining_kwargs = {}
        for key, arg in kwargs.items():
            # TODO(pfackeldey): this should probably do a recursive search (~jax.PyTrees)?
            # single tracer, e.g. nplike.sqrt
            if isinstance(arg, TypeTracerArray):
                tracers.append(arg)
            # multiple tracers, e.g. nplike.broadcast_arrays
            elif isinstance(arg, tuple) and all(
                isinstance(a, TypeTracerArray) for a in arg
            ):
                tracers.extend(arg)
            else:
                remaining_kwargs[key] = arg

        # create a new func_capture
        func_capture = FuncCapture(
            func=func,
            args=(),  # we provide them all as kwargs
            kwargs=remaining_kwargs,
        )

        # get the in_arrays and out_arrays
        in_arrays = tuple(tracers)
        out_arrays = func_capture(*in_arrays)

        # make sure out_arrays is a tuple
        if isinstance(out_arrays, tp.Sequence):
            out_arrays = tuple(out_arrays)
        if not isinstance(out_arrays, tuple):
            out_arrays = (out_arrays,)

        # we only track tracers, any other metadata will be captured anyway
        # TODO: this should do a recursive search (~jax.PyTrees)
        out_arrays = tuple(
            out for out in out_arrays if isinstance(out, TypeTracerArray)
        )

        # create a new trace and return the result,
        # we do that after calling the function to make sure
        # we only add a new main trace if the function call
        # was successful
        with new_main_trace(
            func_capture=func_capture,
            in_arrays=in_arrays,
            out_arrays=out_arrays,  # type: ignore
        ):
            return out_arrays

    return wrapper
