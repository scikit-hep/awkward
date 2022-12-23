# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy

import awkward as ak
from awkward._util import numpy_at_least
from awkward.contents.numpyarray import NumpyArray

# NumPy 1.13.1 introduced NEP13, without which Awkward ufuncs won't work, which
# would be worse than lacking a feature: it would cause unexpected output.
# NumPy 1.17.0 introduced NEP18, which is optional (use ak.* instead of np.*).
if not numpy_at_least("1.13.1"):
    raise ImportError("NumPy 1.13.1 or later required")


def convert_to_array(layout, args, kwargs):
    out = ak.operations.to_numpy(layout, allow_missing=False)
    if args == () and kwargs == {}:
        return out
    else:
        return numpy.array(out, *args, **kwargs)


implemented = {}


def _to_rectilinear(arg):
    if isinstance(arg, tuple):
        nplike = ak._nplikes.nplike_of(*arg)
        return tuple(nplike.to_rectilinear(x) for x in arg)
    else:
        nplike = ak._nplikes.nplike_of(arg)
        return nplike.to_rectilinear(arg)


def array_function(func, types, args, kwargs, behavior):
    function = implemented.get(func)
    if function is None:
        rectilinear_args = tuple(_to_rectilinear(x) for x in args)
        rectilinear_kwargs = {k: _to_rectilinear(v) for k, v in kwargs.items()}
        result = func(*rectilinear_args, **rectilinear_kwargs)
    else:
        result = function(*args, **kwargs)

    # We want the result to be a layout
    out = ak.operations.ak_to_layout._impl(result, allow_record=True, allow_other=True)
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return ak._util.wrap(out, behavior=behavior)
    else:
        return out


def implements(numpy_function):
    def decorator(function):
        implemented[getattr(numpy, numpy_function)] = function
        return function

    return decorator


def _array_ufunc_custom_cast(inputs, behavior):
    args = [
        ak._util.wrap(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]

    nextinputs = []
    for x in args:
        cast_fcn = ak._util.custom_cast(x, behavior)
        if cast_fcn is not None:
            x = cast_fcn(x)
        nextinputs.append(
            ak.operations.to_layout(x, allow_record=True, allow_other=True)
        )
    return nextinputs


def _array_ufunc_adjust(custom, inputs, kwargs, behavior):
    args = [
        ak._util.wrap(x, behavior)
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
    nextinputs = [
        ak._util.wrap(x, behavior)
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else x
        for x in inputs
    ]

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

    return signature


def array_ufunc(ufunc, method, inputs, kwargs):
    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    behavior = ak._util.behavior_of(*inputs)

    inputs = _array_ufunc_custom_cast(inputs, behavior)

    def action(inputs, **ignore):
        signature = _array_ufunc_signature(ufunc, inputs)

        custom = ak._util.overload(behavior, signature)
        if custom is not None:
            return _array_ufunc_adjust(custom, inputs, kwargs, behavior)

        if ufunc is numpy.matmul:
            raise ak._errors.wrap_error(
                NotImplementedError(
                    "matrix multiplication (`@` or `np.matmul`) is not yet implemented for Awkward Arrays"
                )
            )

        if all(
            isinstance(x, NumpyArray) or not isinstance(x, ak.contents.Content)
            for x in inputs
        ):
            backend = ak._backends.backend_of(*inputs)
            nplike = backend.nplike

            # Broadcast parameters against one another
            parameters_factory = ak._broadcasting.intersection_parameters_factory(
                inputs
            )
            (parameters,) = parameters_factory(1)
            if nplike.known_data:
                args = []
                for x in inputs:
                    if isinstance(x, NumpyArray):
                        args.append(x._raw(nplike))
                    else:
                        args.append(x)

                result = backend.apply_ufunc(ufunc, method, args, kwargs)

            else:
                shape = None
                args = []
                for x in inputs:
                    if isinstance(x, NumpyArray):
                        x.data.touch_data()
                        shape = x.shape
                        args.append(numpy.empty((0,) + x.shape[1:], x.dtype))
                    else:
                        args.append(x)
                assert shape is not None
                tmp = getattr(ufunc, method)(*args, **kwargs)
                result = nplike.empty((shape[0],) + tmp.shape[1:], tmp.dtype)
            return (NumpyArray(result, backend=backend, parameters=parameters),)

        for x in inputs:
            if isinstance(x, ak.contents.Content):
                apply_ufunc = ak._util.custom_ufunc(ufunc, x, behavior)
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
            raise ak._errors.wrap_error(
                TypeError(
                    "no {}.{} overloads for custom types: {}".format(
                        type(ufunc).__module__, ufunc.__name__, ", ".join(error_message)
                    )
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

    return ak._util.wrap(out, behavior)


def action_for_matmul(inputs):
    raise ak._errors.wrap_error(NotImplementedError)


try:
    NDArrayOperatorsMixin = numpy.lib.mixins.NDArrayOperatorsMixin

except AttributeError:
    from numpy.core import umath as um

    def _disables_array_ufunc(obj):
        try:
            return obj.__array_ufunc__ is None
        except AttributeError:
            return False

    def _binary_method(ufunc, name):
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(self, other)

        func.__name__ = f"__{name}__"
        return func

    def _reflected_binary_method(ufunc, name):
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(other, self)

        func.__name__ = f"__r{name}__"
        return func

    def _inplace_binary_method(ufunc, name):
        def func(self, other):
            return ufunc(self, other, out=(self,))

        func.__name__ = f"__i{name}__"
        return func

    def _numeric_methods(ufunc, name):
        return (
            _binary_method(ufunc, name),
            _reflected_binary_method(ufunc, name),
            _inplace_binary_method(ufunc, name),
        )

    def _unary_method(ufunc, name):
        def func(self):
            return ufunc(self)

        func.__name__ = f"__{name}__"
        return func

    class NDArrayOperatorsMixin:
        __lt__ = _binary_method(um.less, "lt")
        __le__ = _binary_method(um.less_equal, "le")
        __eq__ = _binary_method(um.equal, "eq")
        __ne__ = _binary_method(um.not_equal, "ne")
        __gt__ = _binary_method(um.greater, "gt")
        __ge__ = _binary_method(um.greater_equal, "ge")

        __add__, __radd__, __iadd__ = _numeric_methods(um.add, "add")
        __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, "sub")
        __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, "mul")
        __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(um.matmul, "matmul")
        __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
            um.true_divide, "truediv"
        )
        __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
            um.floor_divide, "floordiv"
        )
        __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, "mod")
        if hasattr(um, "divmod"):
            __divmod__ = _binary_method(um.divmod, "divmod")
            __rdivmod__ = _reflected_binary_method(um.divmod, "divmod")
        __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, "pow")
        __lshift__, __rlshift__, __ilshift__ = _numeric_methods(um.left_shift, "lshift")
        __rshift__, __rrshift__, __irshift__ = _numeric_methods(
            um.right_shift, "rshift"
        )
        __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, "and")
        __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, "xor")
        __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, "or")

        __neg__ = _unary_method(um.negative, "neg")
        if hasattr(um, "positive"):
            __pos__ = _unary_method(um.positive, "pos")
        __abs__ = _unary_method(um.absolute, "abs")
        __invert__ = _unary_method(um.invert, "invert")
