from __future__ import annotations

import sys

import awkward as ak


def test_argmin_overloaded_reducer():
    awkward_array = ak.Array(
        [[1], [2, 3], [4, 5, 6, 3, 8, 9], [], [], [10]], backend="cuda"
    )

    captured = {}

    # we are looking for the type of the reducer from the awkward._do.reduce (line 236)
    def tracer(frame, event, arg):
        # if awkward._do.reduce is called we start tracing
        if event == "call" and frame.f_code.co_name == "reduce":

            def local_tracer(frame, event, arg):
                # when the variables original_reducer and reducer are initialized we store reducer and return
                if (
                    event == "line"
                    and ("original_reducer" in frame.f_locals)
                    and ("reducer" in frame.f_locals)
                ):
                    captured["reducer"] = frame.f_locals["reducer"]
                return local_tracer

            return local_tracer
        return None

    sys.settrace(tracer)
    ak.argmin(awkward_array, axis=-1)
    sys.settrace(None)

    # finally we check that we are getting the reducer from ak._connect.cuda.reducers and not from awkward._reducers.ArgMin
    assert isinstance(captured["reducer"], ak._connect.cuda.reducers.ArgMin)


def test_argmax_overloaded_reducer():
    awkward_array = ak.Array(
        [[1], [2, 3], [4, 5, 6, 3, 8, 9], [], [], [10]], backend="cuda"
    )

    captured = {}

    # we are looking for the type of the reducer from the awkward._do.reduce (line 236)
    def tracer(frame, event, arg):
        # if awkward._do.reduce is called we start tracing
        if event == "call" and frame.f_code.co_name == "reduce":

            def local_tracer(frame, event, arg):
                # when the variables original_reducer and reducer are initialized we store reducer and return
                if (
                    event == "line"
                    and ("original_reducer" in frame.f_locals)
                    and ("reducer" in frame.f_locals)
                ):
                    captured["reducer"] = frame.f_locals["reducer"]
                return local_tracer

            return local_tracer
        return None

    sys.settrace(tracer)
    ak.argmax(awkward_array, axis=-1)
    sys.settrace(None)

    # finally we check that we are getting the reducer from ak._connect.cuda.reducers and not from awkward._reducers.ArgMax
    assert isinstance(captured["reducer"], ak._connect.cuda.reducers.ArgMax)
