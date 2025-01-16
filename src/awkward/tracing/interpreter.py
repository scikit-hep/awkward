# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import typing as tp
from functools import cached_property
from types import MappingProxyType

from awkward._backends.dispatch import backend_of
from awkward._nplikes.dispatch import nplike_of_obj
from awkward.tracing.core import (
    ArbitraryFunction,
    AwkwardKernelFunction,
    NumpyLikeFunction,
    TraceStack,
)

# None of this is working right now. We should rather think of translating
# our 'TraceStack - DAG' to a 'dask DAG' instead of implementing our own interpreter...


if tp.TYPE_CHECKING:
    from awkward._nplikes.array_like import ArrayLike
    from awkward._nplikes.typetracer import TypeTracerArray


def _get_unique_id(tracer: TypeTracerArray) -> int:
    return id(tracer)


def _get_valid_form_key(tracer: TypeTracerArray) -> str:
    form_key = tracer.form_key
    if form_key is None:
        raise ValueError("form_key must be set")
    return form_key


# let's restrict our input/output to a single buffer mapping of str to ArrayLike
InterpreterInOut = tp.TypeVar("InterpreterInOut", bound=dict[str, ArrayLike])


class Interpreter:  # this should be dask.sync.get if we'd implement the DAG with dask
    """
    This is a simple mini interpreter that replays the trace_stack in order.
    The trace_stack is a list of MainTrace objects, each of them contains
    a FuncCapture object that represents a computation and the input and output
    arrays of that computation. A trace_stack can be seen as a little program,
    which only takes container(s) of buffers as input and returns container(s)
    of buffers as output.

    The benefit of this is that we can avoid all python overhead that comes
    with the computations if we'd run them on highlevel awkward-arrays. Instead,
    we can run the computation on the buffers directly, and skip a lot of layout
    traversals, type checks, etc. - every "metadata" computation that is run already
    in the typetracing step.
    """

    def __init__(self, trace_stack: TraceStack):
        self.trace_stack = trace_stack

    @cached_property
    def lifetimes(self) -> dict[int, int]:
        """
        This is a pre-run step that infers some information we need during the
        actual computation, i.e. until when do we need to keep intermediate buffers?
        """
        # this is a mapping of the unique id of a tracer to the level of the trace
        # until when we need to keep the buffer.
        # Would be nice if we could do this in the typetracing step directly...
        lifetimes: dict[int, int] = {}
        for trace in self.trace_stack:
            for tracer in trace.in_arrays:
                lifetimes[_get_unique_id(tracer)] = trace.level
            for tracer in trace.out_arrays:
                lifetimes[_get_unique_id(tracer)] = trace.level
        return lifetimes

    def run(self, env: InterpreterInOut) -> InterpreterInOut:
        # get the lifetimes of the tracers/buffers
        lifetimes = self.lifetimes

        # the env is a  buffer container that contain all the source buffers needed
        # to run this program; we need to make sure that we never modify them; is this over cautious?
        input_buffer_env = MappingProxyType(env)
        # we need to fill this one with the output buffers
        output_buffer_env: InterpreterInOut = {}

        # this one needs to stay mutable (it's similar to python's locals()/globals())!
        program_state = {}

        def _update_program_state(
            program_state, tracers_and_arrays, level
        ) -> dict[int, ArrayLike]:
            for tracer, array in tracers_and_arrays:
                key = _get_unique_id(tracer)
                program_state[key] = array
                # disable for now, all intermediates stay alive until the end (unless they replace themselves)
                if False and level > lifetimes[key]:
                    # we can delete it if we don't need it anymore
                    # TODO(pfackeldey):
                    #   we need to make sure that we don't delete
                    #   buffers that are supposed to be returned...
                    arr = program_state.pop(key)
                    del arr
            return program_state

        # we need to replay the trace in order, and for every trace we have
        # to look into our buffer container (input_buffer_env)
        # if one of the input arrays is there.
        # if not, they have to come from the previous traces (program_state).
        # otherwise there's a problem!

        def _get_buffer(tracer: TypeTracerArray) -> ArrayLike:
            # 1. check if the buffer is in the input_buffer_env
            # 2. check if the buffer is in the program_state
            # we let it fail if it's not there, that should not happen.
            # TODO(pfackeldey): we should raise a more meaningful error here
            return input_buffer_env.get(
                _get_valid_form_key(tracer), program_state[_get_unique_id(tracer)]
            )

        # the first one is only relying on input arrays from the buffer container
        for main_trace in self.trace_stack:
            in_arrays = tuple(_get_buffer(tracer) for tracer in main_trace.in_arrays)
            if not len(in_arrays) == len(main_trace.in_arrays):
                raise ValueError("Not all input buffers are available")

            func_capture = main_trace.func_capture
            if isinstance(func_capture.func, AwkwardKernelFunction):
                backend = backend_of(in_arrays[0])
                assert all(backend == backend_of(arr) for arr in in_arrays), (
                    "All arrays must be of the same backend"
                )
                func_capture.func = func_capture.func.switch_instance(instance=backend)
                out_arrays = func_capture(*in_arrays)
            elif isinstance(func_capture.func, NumpyLikeFunction):
                nplike = nplike_of_obj(in_arrays[0])
                assert all(nplike == nplike_of_obj(arr) for arr in in_arrays), (
                    "All arrays must be of the same nplike"
                )
                func = func_capture.func.switch_instance(instance=nplike)
                out_arrays = func_capture(*in_arrays)
            else:
                assert isinstance(func_capture.func, ArbitraryFunction)
                # we need to call the function with the nplike and backend
                # as first arguments
                out_arrays = func_capture(*in_arrays)

            # run the first trace
            out_arrays = main_trace.func_capture(*in_arrays)
            if not isinstance(out_arrays, tuple):
                out_arrays = (out_arrays,)

            # update the program_state
            program_state = _update_program_state(
                program_state,
                zip(
                    # tracers (first: in, second: out)
                    main_trace.in_arrays + main_trace.out_arrays,
                    # arrays (first: in, second: out)
                    in_arrays + out_arrays,
                ),
                main_trace.level,
            )
        # if we arrived here, we have the final result in the output_buffer_env,
        # because we don't have any more traces to run. Now it's time to delete all the
        # intermediate buffers that might be still alive.
        program_state.clear()
        return output_buffer_env


# Out = tp.TypeVar("Out")

# def replay_trace(trace_stack: TraceStack, in_arrays: ak.Array, trace_res: Out) -> Out:
#     *_, buffers = ak.to_buffers(in_arrays)

#     interpreter = Interpreter(trace_stack)

#     # run the interpreter
#     out_buffers interpreter.run(buffers)

#     return ak.from_buffers(trace_res.layout.form, next(iter(out_buffers.values())).shape, out_buffers)
