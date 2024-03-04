# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import collections
import functools
import inspect
from collections.abc import Iterable
from itertools import chain

import numpy

import awkward as ak
from awkward._attrs import attrs_of
from awkward._backends.backend import Backend
from awkward._backends.dispatch import backend_of, backend_of_obj, common_backend
from awkward._behavior import (
    behavior_of,
    find_custom_cast,
    find_ufunc,
    find_ufunc_generic,
)
from awkward._categorical import as_hashable
from awkward._layout import wrap_layout
from awkward._nplikes import to_nplike
from awkward._parameters import parameters_intersect
from awkward._regularize import is_non_string_like_iterable
from awkward._typing import Any, Iterator, Mapping
from awkward._util import Sentinel
from awkward.contents.numpyarray import NumpyArray

UNSUPPORTED = Sentinel("UNSUPPORTED", __name__)


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
        backend = backend_of_obj(arg, default=None)
        if backend is not None:
            yield backend
        # Otherwise, traverse into supported sequence types
        elif isinstance(arg, (tuple, list)):
            stack.extend(arg)


def _to_rectilinear(arg, backend: Backend):
    # Is this object something we already associate with a backend?
    arg_backend = backend_of_obj(arg, default=None)
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
            layout = ak.to_layout(
                arg,
                allow_record=False,
                allow_unknown=False,
                primitive_policy="error",
                string_policy="error",
            )
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


def array_function(
    func,
    types,
    args,
    kwargs: dict[str, Any],
    behavior: Mapping | None,
    attrs: Mapping[str, Any] | None = None,
):
    function = implemented.get(func)
    if function is not None:
        return function(*args, **kwargs)
    # Use NumPy's implementation
    else:
        all_arguments = chain(args, kwargs.values())
        unique_backends = frozenset(_find_backends(all_arguments))
        backend = common_backend(unique_backends)

        rectilinear_args = tuple(_to_rectilinear(x, backend) for x in args)
        rectilinear_kwargs = {k: _to_rectilinear(v, backend) for k, v in kwargs.items()}
        result = func(*rectilinear_args, **rectilinear_kwargs)
        # We want the result to be a layout (this will fail for functions returning non-array convertibles)
        out = ak.operations.ak_to_layout._impl(
            result,
            allow_record=True,
            allow_unknown=True,
            none_policy="pass-through",
            regulararray=True,
            use_from_iter=True,
            primitive_policy="pass-through",
            string_policy="pass-through",
        )
        return wrap_layout(out, behavior=behavior, allow_other=True, attrs=attrs)


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


def _array_ufunc_custom_cast(inputs, behavior: Mapping | None, backend):
    args = [
        wrap_layout(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]

    nextinputs = []
    for x in args:
        cast_fcn = find_custom_cast(x, behavior)
        maybe_layout = ak.operations.to_layout(
            x if cast_fcn is None else cast_fcn(x),
            allow_unknown=True,
            primitive_policy="pass-through",
            string_policy="pass-through",
        )
        if isinstance(maybe_layout, (ak.contents.Content, ak.record.Record)):
            maybe_layout = maybe_layout.to_backend(backend)

        nextinputs.append(maybe_layout)
    return nextinputs


def _array_ufunc_adjust(
    custom, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
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


def _array_ufunc_adjust_apply(
    apply_ufunc, ufunc, method, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
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


def _array_ufunc_signature(ufunc, inputs) -> tuple[Any, ...] | None:
    signature = []
    has_seen_nominal_type = False
    for x in inputs:
        if isinstance(x, ak.contents.Content):
            record_name, list_name = x.parameter("__record__"), x.parameter("__list__")
            if record_name is not None:
                signature.append(record_name)
                has_seen_nominal_type = True
            elif list_name is not None:
                signature.append(list_name)
                has_seen_nominal_type = True
            elif isinstance(x, NumpyArray):
                signature.append(x.dtype.type)
            else:
                signature.append(None)
        else:
            signature.append(type(x))

    if has_seen_nominal_type:
        return (ufunc, *signature)
    else:
        return None


def _array_ufunc_categorical(
    ufunc, method: str, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
    if (
        ufunc is numpy.equal
        and len(inputs) == 2
        and isinstance(inputs[0], ak.contents.Content)
        and inputs[0].is_indexed
        and inputs[0].parameter("__array__") == "categorical"
        and isinstance(inputs[1], ak.contents.Content)
        and inputs[1].is_indexed
        and inputs[1].parameter("__array__") == "categorical"
    ):
        assert method == "__call__"
        one, two = inputs

        one_index = numpy.asarray(one.index.data)
        two_index = numpy.asarray(two.index.data)
        one_content = wrap_layout(one.content, behavior)
        two_content = wrap_layout(two.content, behavior)

        if len(one_content) == len(two_content) and ak.operations.all(
            one_content == two_content, axis=None
        ):
            one_mapped = one_index

        else:
            one_list = ak.operations.to_list(one_content)
            two_list = ak.operations.to_list(two_content)
            one_hashable = [as_hashable(x) for x in one_list]
            two_hashable = [as_hashable(x) for x in two_list]
            two_lookup = {x: i for i, x in enumerate(two_hashable)}

            one_to_two = numpy.empty(len(one_hashable) + 1, dtype=numpy.int64)
            for i, x in enumerate(one_hashable):
                one_to_two[i] = two_lookup.get(x, len(two_hashable))
            one_to_two[-1] = -1

            one_mapped = one_to_two[one_index]

        out = one_mapped == two_index
        return (ak.contents.NumpyArray(out),)
    else:
        nextinputs = []
        for x in inputs:
            if isinstance(x, ak.contents.Content) and x.is_indexed:
                nextinputs.append(wrap_layout(x.project(), behavior=behavior))
            else:
                nextinputs.append(wrap_layout(x, behavior=behavior, allow_other=True))

        out = getattr(ufunc, method)(*nextinputs, **kwargs)
        if not isinstance(out, tuple):
            out = (out,)
        return tuple(ak.to_layout(x, allow_unknown=True) for x in out)


def _array_ufunc_string_likes(
    ufunc, method: str, inputs, kwargs: dict[str, Any], behavior: Mapping | None
):
    assert method == "__call__"

    if ufunc not in (numpy.equal, numpy.not_equal) or len(inputs) != 2:
        return

    left, right = inputs

    if isinstance(left, ak.contents.Content) and left.parameter("__array__") in (
        "string",
        "bytestring",
    ):
        left = ak.without_parameters(left, highlevel=False)
    elif isinstance(left, (str, bytes)):
        left = ak.without_parameters([left], highlevel=False)
    else:
        return

    if isinstance(right, ak.contents.Content) and right.parameter("__array__") in (
        "string",
        "bytestring",
    ):
        right = ak.without_parameters(right, highlevel=False)
    elif isinstance(right, (str, bytes)):
        right = ak.without_parameters([right], highlevel=False)
    else:
        return

    left, right = ak.broadcast_arrays(left, right, highlevel=False, depth_limit=1)
    nplike = left.backend.nplike

    # first condition: string lengths must be the same
    left_counts_layout = ak._do.reduce(left, ak._reducers.Count(), axis=-1, mask=False)
    assert left_counts_layout.is_numpy
    right_counts_layout = ak._do.reduce(
        right, ak._reducers.Count(), axis=-1, mask=False
    )
    assert right_counts_layout.is_numpy

    counts1 = nplike.asarray(left_counts_layout.data)
    counts2 = nplike.asarray(right_counts_layout.data)

    out = counts1 == counts2

    # only compare characters in strings that are possibly equal (same length)
    possible = nplike.logical_and(out, counts1)
    possible_counts = counts1[possible]

    if len(possible_counts) > 0:
        onepossible = left[possible]
        twopossible = right[possible]
        reduced = ak.operations.all(
            wrap_layout(onepossible) == wrap_layout(twopossible),
            axis=-1,
            highlevel=False,
        )
        # update same-length strings with a verdict about their characters
        out[possible] = reduced.data

    if ufunc is numpy.not_equal:
        out = nplike.logical_not(out)

    return (ak.contents.NumpyArray(out),)


def array_ufunc(ufunc, method: str, inputs, kwargs: dict[str, Any]):
    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    behavior = behavior_of(*inputs)
    attrs = attrs_of(*inputs)
    backend = backend_of(*inputs, coerce_to_common=True)

    inputs = _array_ufunc_custom_cast(inputs, behavior, backend)

    def action(inputs, **ignore):
        contents = [x for x in inputs if isinstance(x, ak.contents.Content)]
        assert len(contents) >= 1

        signature = _array_ufunc_signature(ufunc, inputs)
        # Should we allow ufunc overloads for this signature?
        if signature is not None:
            # Do we have a custom ufunc (an override of the given ufunc)?
            custom = find_ufunc(behavior, signature)
            if custom is not None:
                return _array_ufunc_adjust(custom, inputs, kwargs, behavior)

        # Do we have any categoricals?
        if any(
            x.is_indexed and x.parameter("__array__") == "categorical" for x in contents
        ):
            out = _array_ufunc_categorical(ufunc, method, inputs, kwargs, behavior)
            if out is not None:
                return out

        # Do we have any strings?
        if any(
            x.is_list and x.parameter("__array__") in ("string", "bytestring")
            for x in contents
        ):
            out = _array_ufunc_string_likes(ufunc, method, inputs, kwargs, behavior)
            if out is not None:
                return out

            # Do we have all-strings? If so, we can't proceed
            if all(
                x.is_list and x.parameter("__array__") in ("string", "bytestring")
                for x in contents
            ):
                raise TypeError(
                    f"{type(ufunc).__module__}.{ufunc.__name__} is not implemented for string types. "
                    "To register an implementation, add a name to these string(s) and register a behavior overload"
                )

        if ufunc is numpy.matmul:
            raise NotImplementedError(
                "matrix multiplication (`@` or `np.matmul`) is not yet implemented for Awkward Arrays"
            )

        # Do we have a custom generic ufunc override (a function that accepts _all_ ufuncs)?
        for x in contents:
            apply_ufunc = find_ufunc_generic(ufunc, x, behavior)
            if apply_ufunc is not None:
                out = _array_ufunc_adjust_apply(
                    apply_ufunc, ufunc, method, inputs, kwargs, behavior
                )
                if out is not None:
                    return out

        if all(
            isinstance(x, NumpyArray) or not isinstance(x, ak.contents.Content)
            for x in inputs
        ):
            # Broadcast parameters against one another
            parameters = functools.reduce(
                parameters_intersect, (c._parameters for c in contents)
            )

            input_args = [x.data if isinstance(x, NumpyArray) else x for x in inputs]
            result = backend.nplike.apply_ufunc(ufunc, method, input_args, kwargs)

            if isinstance(result, tuple):
                return tuple(
                    NumpyArray(x, backend=backend, parameters=parameters)
                    for x in result
                )
            else:
                return (NumpyArray(result, backend=backend, parameters=parameters),)

        # Do we have exclusively nominal types without custom overloads?
        if all(
            x.parameter("__list__") is not None or x.parameter("__record__") is not None
            for x in contents
        ):
            error_message = []
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    if x.parameter("__list__") is not None:
                        error_message.append(x.parameter("__list__"))
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

    out = ak._broadcasting.broadcast_and_apply(
        inputs, action, allow_records=False, function_name=ufunc.__name__
    )

    if len(out) == 1:
        return wrap_layout(out[0], behavior=behavior, attrs=attrs)
    else:
        return tuple(wrap_layout(o, behavior=behavior, attrs=attrs) for o in out)


def action_for_matmul(inputs):
    raise NotImplementedError


def convert_to_array(layout, dtype=None):
    out = ak.operations.to_numpy(layout, allow_missing=False)
    if dtype is None:
        return out
    else:
        return numpy.array(out, dtype=dtype)
