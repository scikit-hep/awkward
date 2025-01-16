import numpy as np

from awkward.tracing.core import trace, new_main_trace, FuncCapture
from awkward._nplikes.typetracer import TypeTracerArray, TypeTracer

nplike = TypeTracer.instance()

### awkward internal
# We need to add this tracing logic to all methods that are called on TypeTracerArrays:
# - nplike.*
# - __add__, __neg__, ...
# - awkward-cpp kernels
# - ?
def sin(tracer_or_array):
    # only trace for TypeTracerArrays
    if isinstance(tracer_or_array, TypeTracerArray):
        # basically a copy
        out_tracer_or_array = tracer_or_array._new(
            tracer_or_array._dtype,
            tracer_or_array._shape,
            tracer_or_array._form_key,
            tracer_or_array._report,
        )
        with new_main_trace(
            func_capture=FuncCapture(nplike, np.sin, (), {}),
            in_arrays=tracer_or_array,
            out_arrays=out_tracer_or_array,
        ):
            return out_tracer_or_array
    # actual computation
    return np.sin(tracer_or_array)

def cos(tracer_or_array):
    # only trace for TypeTracerArrays
    if isinstance(tracer_or_array, TypeTracerArray):
        # basically a copy
        out_tracer_or_array = tracer_or_array._new(
            tracer_or_array._dtype,
            tracer_or_array._shape,
            tracer_or_array._form_key,
            tracer_or_array._report,
        )
        with new_main_trace(
            func_capture=FuncCapture(nplike, np.cos, (), {}),
            in_arrays=tracer_or_array,
            out_arrays=tracer_or_array,
        ):
            return out_tracer_or_array
    # actual computation
    return np.cos(tracer_or_array)


# def fun(x):
#     return cos(sin(x) + cos(x))


### user program
if __name__ == "__main__":
    import awkward as ak

    tracer = ak.to_backend(ak.zip({"x": [1, 2, 3], "y": [4, 5, 6]}), backend="typetracer")


    # this is the example from above
    def fun1(array):
        return cos(sin(array["x"]) + cos(array["y"]))

    out1, dag1 = trace(fun1, tracer)


    # this is with _actual_ awkward core nplike typetracer code
    def fun2(array):
        return np.cos(np.sin(array["x"]) + np.cos(array["y"]))

    out2, dag2 = trace(fun2, tracer)
