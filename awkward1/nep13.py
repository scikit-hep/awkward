# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import numpy

import awkward1.layout
import awkward1._util

def array_ufunc(ufunc, method, inputs, kwargs, classes, functions):
    import awkward1.highlevel

    if method != "__call__" or len(inputs) == 0:
        return NotImplemented
    if "out" in kwargs:
        raise NotImplementedError("in-place operations not supported")

    scalar = all(isinstance(x, (awkward1.highlevel.Record, awkward1.layout.Record)) for x in inputs)

    def unwrap(x):
        if isinstance(x, (awkward1.highlevel.Array, awkward1.highlevel.Record, awkward1.highlevel.FillableArray)):
            return unwrap(x.layout)
        elif isinstance(x, awkward1.layout.FillableArray):
            return unwrap(x.snapshot())
        elif isinstance(x, awkward1.layout.Record):
            return unwrap(x.array)
        else:
            return x

    unknowntypes = (awkward1.layout.EmptyArray,)

    indexedtypes = (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)

    uniontypes = (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)

    regulartypes = (awkward1.layout.RegularArray,)

    listtypes = (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)

    optiontypes = (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)

    recordtypes = (awkward1.layout.RecordArray,)

    def firstof(inputs, types):
        for x in inputs:
            if isinstance(x, types):
                return x
        assert False

    def level(inputs):
        if any(isinstance(x, unknowntypes) for x in inputs):
            return level([x if not isinstance(x, unknowntypes) else awkward1.layout.NumpyArray(numpy.array([], dtype=numpy.int64)) for x in inputs])

        elif any(isinstance(x, indexedtypes) for x in inputs):
            return level([x if not isinstance(x, indexedtypes) else x.project() for x in inputs])

        elif any(isinstance(x, uniontypes) for x in inputs):
            raise NotImplementedError("array_ufunc of UnionArray")

        elif any(isinstance(x, regulartypes) for x in inputs):
            raise NotImplementedError("array_ufunc of RegularArray")

        elif any(isinstance(x, listtypes) for x in inputs):
            first = firstof(inputs, listtypes)
            # FIXME: need a List*Array::compactoffsets() and a *Array::broadcast_to_offsets(offsets)
            offsets = first.offsets
            return awkward1.layout.ListOffsetArray64(offsets, level([x if not isinstance(x, listtypes) else x.content for x in inputs]))

        elif any(isinstance(x, optiontypes) for x in inputs):
            mask = None
            for x in inputs:
                if isinstance(x, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
                    m = numpy.asarray(x.index) < 0
                    if mask is None:
                        mask = m
                    else:
                        numpy.bitwise_or(mask, m, out=mask)
            assert mask is not None
            nextmask = awkward1.layout.Index8(mask.view(numpy.int8))
            index = numpy.full(len(mask), -1, dtype=numpy.int64)
            index[~mask] = numpy.arange(len(mask) - numpy.count_nonzero(mask), dtype=numpy.int64)
            index = awkward1.layout.Index64(index)
            if any(not isinstance(x, optiontypes) for x in inputs):
                nextindex = numpy.arange(len(mask), dtype=numpy.int64)
                nextindex[mask] = -1
                nextindex = awkward1.layout.Index64(nextindex)

            nextinputs = []
            for x in inputs:
                if isinstance(x, optiontypes):
                    nextinputs.append(x.project(nextmask))
                else:
                    nextinputs.append(awkward1.layout.IndexedOptionArray64(nextindex, x).project(nextmask))

            return awkward1.layout.IndexedOptionArray64(index, level(nextinputs))

        elif any(isinstance(x, recordtypes) for x in inputs):
            raise NotImplementedError("array_ufunc of RecordArray")

        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)
            return awkward1.layout.NumpyArray(result)

    out = awkward1._util.wrap(level([unwrap(x) for x in inputs]), classes, functions)
    if scalar:
        return out[0]
    else:
        return out

try:
    NDArrayOperatorsMixin = numpy.lib.mixins.NDArrayOperatorsMixin

except AttributeError:
    from numpy.core import umath as um

    def _disables_array_ufunc(obj):
        """True when __array_ufunc__ is set to None."""
        try:
            return obj.__array_ufunc__ is None
        except AttributeError:
            return False

    def _binary_method(ufunc, name):
        """Implement a forward binary method with a ufunc, e.g., __add__."""
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(self, other)
        func.__name__ = "__{}__".format(name)
        return func

    def _reflected_binary_method(ufunc, name):
        """Implement a reflected binary method with a ufunc, e.g., __radd__."""
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(other, self)
        func.__name__ = "__r{}__".format(name)
        return func

    def _inplace_binary_method(ufunc, name):
        """Implement an in-place binary method with a ufunc, e.g., __iadd__."""
        def func(self, other):
            return ufunc(self, other, out=(self,))
        func.__name__ = "__i{}__".format(name)
        return func

    def _numeric_methods(ufunc, name):
        """Implement forward, reflected and inplace binary methods with a ufunc."""
        return (_binary_method(ufunc, name),
                _reflected_binary_method(ufunc, name),
                _inplace_binary_method(ufunc, name))

    def _unary_method(ufunc, name):
        """Implement a unary special method with a ufunc."""
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

        # numeric methods
        __add__, __radd__, __iadd__ = _numeric_methods(um.add, "add")
        __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, "sub")
        __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, "mul")
        if sys.version_info.major < 3:
            # Python 3 uses only __truediv__ and __floordiv__
            __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide, "div")
        __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
            um.true_divide, "truediv")
        __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
            um.floor_divide, "floordiv")
        __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, "mod")
        if hasattr(um, "divmod"):
            __divmod__ = _binary_method(um.divmod, "divmod")
            __rdivmod__ = _reflected_binary_method(um.divmod, "divmod")
        # __idivmod__ does not exist
        # TODO: handle the optional third argument for __pow__?
        __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, "pow")
        __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
            um.left_shift, "lshift")
        __rshift__, __rrshift__, __irshift__ = _numeric_methods(
            um.right_shift, "rshift")
        __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, "and")
        __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, "xor")
        __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, "or")

        # unary methods
        __neg__ = _unary_method(um.negative, "neg")
        if hasattr(um, "positive"):
            __pos__ = _unary_method(um.positive, "pos")
            __abs__ = _unary_method(um.absolute, "abs")
            __invert__ = _unary_method(um.invert, "invert")
