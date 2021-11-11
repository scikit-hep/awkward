# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import sys

import numpy

import awkward as ak
from awkward._v2.contents.numpyarray import NumpyArray


# def convert_to_array(layout, args, kwargs):
#     out = ak._v2.operations.convert.to_numpy(layout, allow_missing=False)
#     if args == () and kwargs == {}:
#         return out
#     else:
#         return numpy.array(out, *args, **kwargs)


# implemented = {}


# def _to_rectilinear(arg):
#     if isinstance(
#         arg,
#         (
#             ak.Array,
#             ak.Record,
#             ak.ArrayBuilder,
#             ak._v2.contents.Content,
#             ak._v2.record.Record,
#             ak.layout.ArrayBuilder,
#             ak.layout.LayoutBuilder32,
#             ak.layout.LayoutBuilder64,
#         ),
#     ):
#         nplike = ak.nplike.of(arg)
#         return nplike.to_rectilinear(arg)
#     else:
#         return arg


# def array_function(func, types, args, kwargs):
#     function = implemented.get(func)
#     if function is None:
#         args = tuple(_to_rectilinear(x) for x in args)
#         kwargs = dict((k, _to_rectilinear(v)) for k, v in kwargs.items())
#         out = func(*args, **kwargs)
#         nplike = ak.nplike.of(out)
#         if isinstance(out, nplike.ndarray) and len(out.shape) != 0:
#             return ak.Array(out)
#         else:
#             return out
#     else:
#         return function(*args, **kwargs)


# def implements(numpy_function):
#     def decorator(function):
#         implemented[getattr(numpy, numpy_function)] = function
#         return function

#     return decorator


def _array_ufunc_custom_cast(inputs, behavior):
    nextinputs = []
    for x in inputs:
        cast_fcn = ak._v2._util.custom_cast(x, behavior)
        if cast_fcn is not None:
            x = cast_fcn(x)
        nextinputs.append(
            ak._v2.operations.convert.to_layout(x, allow_record=True, allow_other=True)
        )
    return nextinputs


def _array_ufunc_adjust(custom, inputs, kwargs, behavior):
    args = [
        ak._v2._util.wrap(x, behavior)
        if isinstance(x, (ak._v2.contents.Content, ak._v2.record.Record))
        else x
        for x in inputs
    ]
    out = custom(*args, **kwargs)
    if not isinstance(out, tuple):
        out = (out,)

    return tuple(
        x.layout
        if isinstance(x, (ak._v2.highlevel.Array, ak._v2.highlevel.Record))
        else x
        for x in out
    )


def _array_ufunc_adjust_apply(apply_ufunc, ufunc, method, inputs, kwargs, behavior):
    nextinputs = [
        ak._v2._util.wrap(x, behavior)
        if isinstance(x, (ak._v2.contents.Content, ak._v2.record.Record))
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
            x.layout
            if isinstance(x, (ak._v2.highlevel.Array, ak._v2.highlevel.Record))
            else x
            for x in out
        )


def _array_ufunc_signature(ufunc, inputs):
    signature = [ufunc]
    for x in inputs:
        if isinstance(x, ak._v2.contents.Content):
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


def _array_ufunc_deregulate(inputs):
    nextinputs = []
    for x in inputs:
        if isinstance(x, ak._v2.contents.RegularArray):
            y = x.maybe_toNumpyArray()
            if y is not None:
                nextinputs.append(y)
            else:
                nextinputs.append(x)
        else:
            nextinputs.append(x)

    return nextinputs


def array_ufunc(ufunc, method, inputs, kwargs):
    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    behavior = ak._v2._util.behavior_of(*inputs)

    inputs = _array_ufunc_custom_cast(inputs, behavior)

    def action(inputs, **ignore):
        signature = _array_ufunc_signature(ufunc, inputs)

        custom = ak._v2._util.overload(behavior, signature)
        if custom is not None:
            return _array_ufunc_adjust(custom, inputs, kwargs, behavior)

        if ufunc is numpy.matmul:
            custom_matmul = action_for_matmul(inputs)
            if custom_matmul is not None:
                return custom_matmul()

        inputs = _array_ufunc_deregulate(inputs)

        if all(
            isinstance(x, NumpyArray) or not isinstance(x, ak._v2.contents.Content)
            for x in inputs
        ):
            nplike = ak.nplike.of(*inputs)
            args = [x.to(nplike) if isinstance(x, NumpyArray) else x for x in inputs]
            result = getattr(ufunc, method)(*args, **kwargs)
            return (NumpyArray(result, nplike=nplike),)

        for x in inputs:
            if isinstance(x, ak._v2.contents.Content):
                apply_ufunc = ak._v2._util.custom_ufunc(ufunc, x, behavior)
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
            if isinstance(x, ak._v2.contents.Content)
        ):
            error_message = []
            for x in inputs:
                if isinstance(x, ak._v2.contents.Content):
                    if x.parameter("__array__") is not None:
                        error_message.append(x.parameter("__array__"))
                    elif x.parameter("__record__") is not None:
                        error_message.append(x.parameter("__record__"))
                    else:
                        error_message.append(type(x).__name__)
                else:
                    error_message.append(type(x).__name__)
            raise TypeError(
                "no {0}.{1} overloads for custom types: {2}".format(
                    type(ufunc).__module__, ufunc.__name__, ", ".join(error_message)
                )
            )

        return None

    out = ak._v2._broadcasting.broadcast_and_apply(
        inputs, action, behavior, allow_records=False, function_name=ufunc.__name__
    )
    assert isinstance(out, tuple) and len(out) == 1
    return ak._v2._util.wrap(out[0], behavior)


# def matmul_for_numba(lefts, rights, dtype):
#     total_outer = 0
#     total_inner = 0
#     total_content = 0

#     for A, B in zip(lefts, rights):
#         first = -1
#         for Ai in A:
#             if first == -1:
#                 first = len(Ai)
#             elif first != len(Ai):
#                 raise ValueError(
#                     "one of the left matrices in np.matmul is not rectangular"
#                 )
#         if first == -1:
#             first = 0
#         rowsA = len(A)
#         colsA = first

#         first = -1
#         for Bi in B:
#             if first == -1:
#                 first = len(Bi)
#             elif first != len(Bi):
#                 raise ValueError(
#                     "one of the right matrices in np.matmul is not rectangular"
#                 )
#         if first == -1:
#             first = 0
#         rowsB = len(B)
#         colsB = first

#         if colsA != rowsB:
#             raise ValueError(
#                 u"one of the pairs of matrices in np.matmul do not match shape: "
#                 u"(n \u00d7 k) @ (k \u00d7 m)"
#             )

#         total_outer += 1
#         total_inner += rowsA
#         total_content += rowsA * colsB

#     outer = numpy.empty(total_outer + 1, numpy.int64)
#     inner = numpy.empty(total_inner + 1, numpy.int64)
#     content = numpy.zeros(total_content, dtype)

#     outer[0] = 0
#     inner[0] = 0
#     outer_i = 1
#     inner_i = 1
#     content_i = 0
#     for A, B in zip(lefts, rights):
#         rows = len(A)
#         cols = 0
#         if len(B) > 0:
#             cols = len(B[0])
#         mids = 0
#         if len(A) > 0:
#             mids = len(A[0])

#         for i in range(rows):
#             for j in range(cols):
#                 for v in range(mids):
#                     pos = content_i + i * cols + j
#                     content[pos] += A[i][v] * B[v][j]

#         outer[outer_i] = outer[outer_i - 1] + rows
#         outer_i += 1
#         for _ in range(rows):
#             inner[inner_i] = inner[inner_i - 1] + cols
#             inner_i += 1
#         content_i += rows * cols

#     return outer, inner, content


# matmul_for_numba.numbafied = None


def action_for_matmul(inputs):
    raise NotImplementedError


# def action_for_matmul(inputs):
#     inputs = [
#         ak._v2._util.recursively_apply(
#             x, (lambda _: _), pass_depth=False, numpy_to_regular=True
#         )
#         if isinstance(x, (ak._v2.contents.Content, ak._v2.record.Record))
#         else x
#         for x in inputs
#     ]

#     if len(inputs) == 2 and all(
#         isinstance(x, ak._v2._util.listtypes)
#         and isinstance(x.content, ak._v2._util.listtypes)
#         and isinstance(x.content.content, NumpyArray)
#         for x in inputs
#     ):
#         ak._v2._connect.numba.register_and_check()
#         import numba

#         if matmul_for_numba.numbafied is None:
#             matmul_for_numba.numbafied = numba.njit(matmul_for_numba)

#         lefts = ak._v2.highlevel.Array(inputs[0])
#         rights = ak._v2.highlevel.Array(inputs[1])
#         dtype = numpy.asarray(lefts[0:0, 0:0, 0:0] + rights[0:0, 0:0, 0:0]).dtype

#         outer, inner, content = matmul_for_numba.numbafied(lefts, rights, dtype)

#         return lambda: (
#             ak._v2.contents.ListOffsetArray64(
#                 ak._v2.index.Index64(outer),
#                 ak._v2.contents.ListOffsetArray64(
#                     ak._v2.index.Index64(inner),
#                     NumpyArray(content),
#                 ),
#             ),
#         )

#     else:
#         return None


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
        __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(um.matmul, "matmul")
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
