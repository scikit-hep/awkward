# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import numpy

import awkward1.layout
import awkward1.operations.convert
import awkward1._util
import awkward1.nplike


def convert_to_array(layout, args, kwargs):
    out = awkward1.operations.convert.to_numpy(layout, allow_missing=False)
    if args == () and kwargs == {}:
        return out
    else:
        return numpy.array(out, *args, **kwargs)


implemented = {}


def array_function(func, types, args, kwargs):
    function = implemented.get(func)
    if function is None:
        return NotImplemented
    else:
        return function(*args, **kwargs)


def implements(numpy_function):
    def decorator(function):
        implemented[getattr(numpy, numpy_function)] = function
        return function
    return decorator


def array_ufunc(ufunc, method, inputs, kwargs):
    import awkward1.highlevel

    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    behavior = awkward1._util.behaviorof(*inputs)
    inputs = [
        awkward1.operations.convert.to_layout(x, allow_record=True, allow_other=True)
        for x in inputs
    ]

    def adjust(custom, inputs, kwargs):
        args = [
            awkward1._util.wrap(x, behavior)
            if isinstance(x, (awkward1.layout.Content, awkward1.layout.Record))
            else x
            for x in inputs
        ]
        out = custom(*args, **kwargs)
        if not isinstance(out, tuple):
            out = (out,)

        return tuple(
            x.layout
            if isinstance(x, (awkward1.highlevel.Array, awkward1.highlevel.Record))
            else x
            for x in out
        )

    def adjust_apply_ufunc(apply_ufunc, ufunc, method, inputs, kwargs):
        nextinputs = [
            awkward1._util.wrap(x, behavior)
            if isinstance(x, (awkward1.layout.Content, awkward1.layout.Record))
            else x
            for x in inputs
        ]

        out = apply_ufunc(ufunc, method, nextinputs, kwargs)

        if out is NotImplemented:
            return None
        else:
            if not isinstance(out, tuple):
                out = (out,)
            out = tuple(
                x.layout
                if isinstance(x, (awkward1.highlevel.Array, awkward1.highlevel.Record))
                else x
                for x in out
            )
            return lambda: out

    def getfunction(inputs, depth):
        signature = [ufunc]
        for x in inputs:
            if isinstance(x, awkward1.layout.Content):
                record = x.parameters.get("__record__")
                array = x.parameters.get("__array__")
                if record is not None:
                    signature.append(record)
                elif array is not None:
                    signature.append(array)
                elif isinstance(x, awkward1.layout.NumpyArray):
                    signature.append(awkward1.nplike.of(x).asarray(x).dtype.type)
                else:
                    signature.append(None)
            else:
                signature.append(type(x))

        custom = awkward1._util.overload(behavior, signature)
        if custom is not None:
            return lambda: adjust(custom, inputs, kwargs)

        if all(
            isinstance(x, awkward1.layout.NumpyArray)
            or not isinstance(
                x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
            )
            for x in inputs
        ):
            nplike = awkward1.nplike.of(*inputs)
            result = getattr(ufunc, method)(
                *[nplike.asarray(x) for x in inputs], **kwargs
            )
            return lambda: (awkward1.layout.NumpyArray(result),)

        for x in inputs:
            if isinstance(x, awkward1.layout.Content):
                chained_behavior = awkward1._util.Behavior(awkward1.behavior, behavior)
                apply_ufunc = chained_behavior[numpy.ufunc, x.parameter("__array__")]
                if apply_ufunc is not None:
                    out = adjust_apply_ufunc(
                        apply_ufunc, ufunc, method, inputs, kwargs
                    )
                    if out is not None:
                        return out
                apply_ufunc = chained_behavior[numpy.ufunc, x.parameter("__record__")]
                if apply_ufunc is not None:
                    out = adjust_apply_ufunc(
                        apply_ufunc, ufunc, method, inputs, kwargs
                    )
                    if out is not None:
                        return out

        if all(
            x.parameter("__array__") is not None
            or x.parameter("__record__") is not None
            for x in inputs if isinstance(x, awkward1.layout.Content)
        ):
            custom_types = []
            for x in inputs:
                if isinstance(x, awkward1.layout.Content):
                    if x.parameter("__array__") is not None:
                        custom_types.append(x.parameter("__array__"))
                    elif x.parameter("__record__") is not None:
                        custom_types.append(x.parameter("__record__"))
                    else:
                        custom_types.append(type(x).__name__)
                else:
                    custom_types.append(type(x).__name__)
            exception = ValueError(
                "no overloads for custom types: {0}({1})".format(
                    ufunc.__name__,
                    ", ".join(custom_types),
                )
                + awkward1._util.exception_suffix(__file__)
            )
            awkward1._util.deprecate(exception, "1.0.0", date="2020-12-01")

        return None

    out = awkward1._util.broadcast_and_apply(
        inputs, getfunction, behavior, allow_records=False
    )
    assert isinstance(out, tuple) and len(out) == 1
    return awkward1._util.wrap(out[0], behavior)


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

        func.__name__ = "__{}__".format(name)
        return func

    def _reflected_binary_method(ufunc, name):
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(other, self)

        func.__name__ = "__r{}__".format(name)
        return func

    def _inplace_binary_method(ufunc, name):
        def func(self, other):
            return ufunc(self, other, out=(self,))

        func.__name__ = "__i{}__".format(name)
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

        func.__name__ = "__{}__".format(name)
        return func

    class NDArrayOperatorsMixin(object):
        __lt__ = _binary_method(um.less, "lt")
        __le__ = _binary_method(um.less_equal, "le")
        __eq__ = _binary_method(um.equal, "eq")
        __ne__ = _binary_method(um.not_equal, "ne")
        __gt__ = _binary_method(um.greater, "gt")
        __ge__ = _binary_method(um.greater_equal, "ge")

        __add__, __radd__, __iadd__ = _numeric_methods(um.add, "add")
        __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, "sub")
        __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, "mul")
        if sys.version_info.major < 3:
            __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide, "div")
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
        __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
            um.left_shift, "lshift"
        )
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
