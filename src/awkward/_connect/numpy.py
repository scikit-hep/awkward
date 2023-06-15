# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import collections
import functools
import inspect
from collections.abc import Iterable
from itertools import chain

import numpy

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import backend_of, common_backend
from awkward._behavior import (
    behavior_of,
    find_custom_cast,
    find_ufunc,
    find_ufunc_generic,
)
from awkward._layout import wrap_layout
from awkward._nplikes import to_nplike
from awkward._regularize import is_non_string_like_iterable
from awkward._typing import Iterator
from awkward._util import Sentinel, numpy_at_least
from awkward.contents.numpyarray import NumpyArray

# NumPy 1.13.1 introduced NEP13, without which Awkward ufuncs won't work, which
# would be worse than lacking a feature: it would cause unexpected output.
# NumPy 1.17.0 introduced NEP18, which is optional (use ak.* instead of np.*).
if not numpy_at_least("1.13.1"):
    raise ImportError("NumPy 1.13.1 or later required")


UNSUPPORTED = Sentinel("UNSUPPORTED", __name__)


def convert_to_array(layout, args, kwargs):
    out = ak.operations.to_numpy(layout, allow_missing=False)
    if args == () and kwargs == {}:
        return out
    else:
        return numpy.array(out, *args, **kwargs)


implemented = {}


def _find_backends(args: Iterable) -> Iterator[Backend]:
    """
    Args:
        args: iterable of objects to visit

    Yields the encountered backends of layout / array-like arguments encountered
    in the argument list.
    """
    stack = collections.deque(args)
    while stack:
        arg = stack.popleft()
        # If the argument declares a backend, easy!
        backend = backend_of(arg, default=None)
        if backend is not None:
            yield backend
        # Otherwise, traverse into supported sequence types
        elif isinstance(arg, (tuple, list)):
            stack.extend(arg)


def _to_rectilinear(arg, backend: Backend):
    # Is this object something we already associate with a backend?
    arg_backend = backend_of(arg, default=None)
    if arg_backend is not None:
        arg_nplike = arg_backend.nplike
        # Is this argument already in a backend-supported form?
        if arg_nplike.is_own_array(arg):
            # Convert to the appropriate nplike
            return to_nplike(
                arg_nplike.asarray(arg), backend.nplike, from_nplike=arg_nplike
            )
        # Otherwise, cast to layout and convert
        else:
            layout = ak.to_layout(arg, allow_record=False, allow_other=False)
            return layout.to_backend(backend).to_backend_array(allow_missing=True)
    elif isinstance(arg, tuple):
        return tuple(_to_rectilinear(x, backend) for x in arg)
    elif isinstance(arg, list):
        return [_to_rectilinear(x, backend) for x in arg]
    elif is_non_string_like_iterable(arg):
        raise TypeError(
            f"encountered an unsupported iterable value {arg!r} whilst converting arguments to NumPy-friendly "
            f"types. If this argument should be supported, please file a bug report."
        )
    else:
        return arg


def _array_function_no_impl(func, types, args, kwargs, behavior):
    backend = common_backend(_find_backends(chain(args, kwargs.values())))

    rectilinear_args = tuple(_to_rectilinear(x, backend) for x in args)
    rectilinear_kwargs = {k: _to_rectilinear(v, backend) for k, v in kwargs.items()}
    result = func(*rectilinear_args, **rectilinear_kwargs)
    # We want the result to be a layout (this will fail for functions returning non-array convertibles)
    out = ak.operations.ak_to_layout._impl(
        result, allow_record=True, allow_other=True, regulararray=True
    )
    return wrap_layout(out, behavior=behavior, allow_other=True)


def array_function(func, types, args, kwargs, behavior):
    function = implemented.get(func)
    # Use NumPy's implementation
    if function is None:
        return _array_function_no_impl(func, types, args, kwargs, behavior)
    else:
        return function(*args, **kwargs)


def implements(numpy_function):
    def decorator(function):
        signature = inspect.signature(function)
        unsupported_names = {
            p.name for p in signature.parameters.values() if p.default is UNSUPPORTED
        }

        @functools.wraps(function)
        def ensure_valid_args(*args, **kwargs):
            parameters = signature.bind(*args, **kwargs)
            provided_invalid_names = parameters.arguments.keys() & unsupported_names
            if provided_invalid_names:
                names = ", ".join(provided_invalid_names)
                raise TypeError(
                    f"Awkward NEP-18 overload was provided with unsupported argument(s): {names}"
                )
            return function(*args, **kwargs)

        implemented[getattr(numpy, numpy_function)] = ensure_valid_args
        return function

    return decorator


def _array_ufunc_custom_cast(inputs, behavior, backend):
    args = [
        wrap_layout(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]

    nextinputs = []
    for x in args:
        cast_fcn = find_custom_cast(x, behavior)
        if cast_fcn is not None:
            x = cast_fcn(x)
        maybe_layout = ak.operations.to_layout(x, allow_record=True, allow_other=True)
        if isinstance(maybe_layout, (ak.contents.Content, ak.record.Record)):
            maybe_layout = maybe_layout.to_backend(backend)

        nextinputs.append(maybe_layout)
    return nextinputs


def _array_ufunc_adjust(custom, inputs, kwargs, behavior):
    args = [
        wrap_layout(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]
    out = custom(*args, **kwargs)
    if not isinstance(out, tuple):
        out = (out,)

    return tuple(
        x.layout if isinstance(x, (ak.highlevel.Array, ak.highlevel.Record)) else x
        for x in out
    )


def _array_ufunc_adjust_apply(apply_ufunc, ufunc, method, inputs, kwargs, behavior):
    nextinputs = [wrap_layout(x, behavior, allow_other=True) for x in inputs]
    out = apply_ufunc(ufunc, method, nextinputs, kwargs)

    if out is NotImplemented:
        return None
    else:
        if not isinstance(out, tuple):
            out = (out,)
        return tuple(
            x.layout if isinstance(x, (ak.highlevel.Array, ak.highlevel.Record)) else x
            for x in out
        )


def _array_ufunc_signature(ufunc, inputs):
    signature = [ufunc]
    for x in inputs:
        if isinstance(x, ak.contents.Content):
            record, array = x.parameter("__record__"), x.parameter("__array__")
            if record is not None:
                signature.append(record)
            elif array is not None:
                signature.append(array)
            elif isinstance(x, NumpyArray):
                signature.append(x.dtype.type)
            else:
                signature.append(None)
        else:
            signature.append(type(x))

    return tuple(signature)


def array_ufunc(ufunc, method, inputs, kwargs):
    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    behavior = behavior_of(*inputs)
    backend = backend_of(*inputs)

    inputs = _array_ufunc_custom_cast(inputs, behavior, backend)

    def action(inputs, **ignore):
        signature = _array_ufunc_signature(ufunc, inputs)
        custom = find_ufunc(behavior, signature)
        # Do we have a custom ufunc (an override of the given ufunc)?
        if custom is not None:
            return _array_ufunc_adjust(custom, inputs, kwargs, behavior)

        if ufunc is numpy.matmul:
            raise NotImplementedError(
                "matrix multiplication (`@` or `np.matmul`) is not yet implemented for Awkward Arrays"
            )

        if all(
            isinstance(x, NumpyArray) or not isinstance(x, ak.contents.Content)
            for x in inputs
        ):
            nplike = backend.nplike

            # Broadcast parameters against one another
            parameters_factory = ak._broadcasting.intersection_parameters_factory(
                inputs
            )
            (parameters,) = parameters_factory(1)

            args = []
            for x in inputs:
                if isinstance(x, NumpyArray):
                    args.append(x._raw(nplike))
                else:
                    args.append(x)

            # Give backend a chance to change the ufunc implementation
            impl = backend.prepare_ufunc(ufunc)

            # Invoke ufunc
            result = impl(*args, **kwargs)

            return (NumpyArray(result, backend=backend, parameters=parameters),)

        # Do we have a custom generic ufunc override (a function that accepts _all_ ufuncs)?
        for x in inputs:
            if isinstance(x, ak.contents.Content):
                apply_ufunc = find_ufunc_generic(ufunc, x, behavior)
                if apply_ufunc is not None:
                    out = _array_ufunc_adjust_apply(
                        apply_ufunc, ufunc, method, inputs, kwargs, behavior
                    )
                    if out is not None:
                        return out

        if all(
            x.parameter("__array__") is not None
            or x.parameter("__record__") is not None
            for x in inputs
            if isinstance(x, ak.contents.Content)
        ):
            error_message = []
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    if x.parameter("__array__") is not None:
                        error_message.append(x.parameter("__array__"))
                    elif x.parameter("__record__") is not None:
                        error_message.append(x.parameter("__record__"))
                    else:
                        error_message.append(type(x).__name__)
                else:
                    error_message.append(type(x).__name__)
            raise TypeError(
                "no {}.{} overloads for custom types: {}".format(
                    type(ufunc).__module__, ufunc.__name__, ", ".join(error_message)
                )
            )

        return None

    if sum(int(isinstance(x, ak.contents.Content)) for x in inputs) == 1:
        where = None
        for i, x in enumerate(inputs):
            if isinstance(x, ak.contents.Content):
                where = i
                break
        assert where is not None

        nextinputs = list(inputs)

        def unary_action(layout, **ignore):
            nextinputs[where] = layout
            result = action(tuple(nextinputs), **ignore)
            if result is None:
                return None
            else:
                assert isinstance(result, tuple) and len(result) == 1
                return result[0]

        out = ak._do.recursively_apply(
            inputs[where],
            unary_action,
            behavior,
            function_name=ufunc.__name__,
            allow_records=False,
        )

    else:
        out = ak._broadcasting.broadcast_and_apply(
            inputs, action, behavior, allow_records=False, function_name=ufunc.__name__
        )
        assert isinstance(out, tuple) and len(out) == 1
        out = out[0]

    return wrap_layout(out, behavior)


def action_for_matmul(inputs):
    raise NotImplementedError
