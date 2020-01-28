# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1.layout
import awkward1.operations.convert
import awkward1._util

def convert_to_array(layout, args, kwargs):
    out = awkward1.operations.convert.tonumpy(layout)
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
        implemented[numpy_function] = function
        return function
    return decorator

def array_ufunc(ufunc, method, inputs, kwargs, classes, functions):
    import awkward1.highlevel

    if method != "__call__" or len(inputs) == 0 or "out" in kwargs:
        return NotImplemented

    def unwrap(x):
        if isinstance(x, (awkward1.highlevel.Array, awkward1.highlevel.Record)):
            return x.layout
        elif isinstance(x, awkward1.highlevel.FillableArray):
            return x.snapshot().layout
        elif isinstance(x, awkward1.layout.FillableArray):
            return x.snapshot()
        elif isinstance(x, (awkward1.layout.Content, awkward1.layout.Record)):
            return x
        elif isinstance(x, numpy.ndarray):
            if issubclass(x.dtype.type, numpy.number):
                return awkward1.highlevel.Array(x).layout
            else:
                raise ValueError("numpy.ndarray with {0} cannot be used in {1}".format(repr(x.dtype), ufunc))
        elif isinstance(x, Iterable):
            return unwrap(numpy.array(x))
        else:
            return x

    def checklength(inputs):
        length = len(inputs[0])
        for x in inputs[1:]:
            if len(x) != length:
                raise ValueError("cannot broadcast {0} of length {1} with {2} of length {3}".format(type(inputs[0]).__name__, length, type(x).__name__, len(x)))

    def apply(inputs):
        # handle implicit right-broadcasting (i.e. NumPy-like)
        if any(isinstance(x, awkward1._util.listtypes) for x in inputs):
            maxdepth = max(x.purelist_depth for x in inputs if isinstance(x, awkward1.layout.Content))
            if maxdepth > 0 and all(x.purelist_isregular for x in inputs if isinstance(x, awkward1.layout.Content)):
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1.layout.Content):
                        while x.purelist_depth < maxdepth:
                            x = awkward1.layout.RegularArray(x, 1)
                    nextinputs.append(x)
                if any(x is not y for x, y in zip(inputs, nextinputs)):
                    return apply(nextinputs)

        # now all lengths must agree
        checklength([x for x in inputs if isinstance(x, awkward1.layout.Content)])

        signature = (ufunc,) + tuple(x.parameters.get("__class__") if isinstance(x, awkward1.layout.Content) else type(x) for x in inputs)
        custom = awkward1._util.regular_functions(functions).get(signature)

        # the rest of this is one switch statement
        if custom is not None:
            return custom(*inputs, **kwargs)

        elif any(isinstance(x, awkward1._util.unknowntypes) for x in inputs):
            return apply([x if not isinstance(x, awkward1._util.unknowntypes) else awkward1.layout.NumpyArray(numpy.array([], dtype=numpy.int64)) for x in inputs])

        elif any(isinstance(x, awkward1.layout.NumpyArray) and x.ndim > 1 for x in inputs):
            return apply([x if not (isinstance(x, awkward1.layout.NumpyArray) and x.ndim > 1) else x.regularize_shape() for x in inputs])

        elif any(isinstance(x, awkward1._util.indexedtypes) for x in inputs):
            return apply([x if not isinstance(x, awkward1._util.indexedtypes) else x.project() for x in inputs])

        elif any(isinstance(x, awkward1._util.uniontypes) for x in inputs):
            tagslist = []
            length = None
            for x in inputs:
                if isinstance(x, awkward1._util.uniontypes):
                    tagslist.append(numpy.asarray(x.tags))
                    if length is None:
                        length = len(tagslist[-1])
                    elif length != len(tagslist[-1]):
                        raise ValueError("cannot broadcast UnionArray of length {0} with UnionArray of length {1}".format(length, len(tagslist[-1])))

            combos = numpy.stack(tagslist, axis=-1)
            combos = combos.view([(str(i), combos.dtype) for i in range(len(tagslist))]).reshape(length)

            tags = numpy.empty(length, dtype=numpy.int8)
            index = numpy.empty(length, dtype=numpy.int64)
            contents = []
            for tag, combo in enumerate(numpy.unique(combos)):
                mask = (combos == combo)
                tags[mask] = tag
                index[mask] = numpy.arange(numpy.count_nonzero(mask))
                nextinputs = []
                for i, x in enumerate(inputs):
                    if isinstance(x, awkward1._util.uniontypes):
                        nextinputs.append(x[mask].project(combo[str(i)]))
                    elif isinstance(x, awkward1.layout.Content):
                        nextinputs.append(x[mask])
                    else:
                        nextinputs.append(x)
                contents.append(apply(nextinputs))

            tags = awkward1.layout.Index8(tags)
            index = awkward1.layout.Index64(index)
            return awkward1.layout.UnionArray8_64(tags, index, contents)

        elif any(isinstance(x, awkward1._util.optiontypes) for x in inputs):
            mask = None
            for x in inputs:
                if isinstance(x, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
                    m = numpy.asarray(x.index) < 0
                    if mask is None:
                        mask = m
                    else:
                        numpy.bitwise_or(mask, m, out=mask)

            nextmask = awkward1.layout.Index8(mask.view(numpy.int8))
            index = numpy.full(len(mask), -1, dtype=numpy.int64)
            index[~mask] = numpy.arange(len(mask) - numpy.count_nonzero(mask), dtype=numpy.int64)
            index = awkward1.layout.Index64(index)
            if any(not isinstance(x, awkward1._util.optiontypes) for x in inputs):
                nextindex = numpy.arange(len(mask), dtype=numpy.int64)
                nextindex[mask] = -1
                nextindex = awkward1.layout.Index64(nextindex)

            nextinputs = []
            for x in inputs:
                if isinstance(x, awkward1._util.optiontypes):
                    nextinputs.append(x.project(nextmask))
                else:
                    nextinputs.append(awkward1.layout.IndexedOptionArray64(nextindex, x).project(nextmask))

            return awkward1.layout.IndexedOptionArray64(index, apply(nextinputs))

        elif any(isinstance(x, awkward1._util.listtypes) for x in inputs):
            if all(isinstance(x, awkward1.layout.RegularArray) or not isinstance(x, awkward1._util.listtypes) for x in inputs):
                maxsize = max([x.size for x in inputs if isinstance(x, awkward1.layout.RegularArray)])
                for x in inputs:
                    if isinstance(x, awkward1.layout.RegularArray):
                        if maxsize > 1 and x.size == 1:
                            tmpindex = awkward1.layout.Index64(numpy.repeat(numpy.arange(len(x), dtype=numpy.int64), maxsize))
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1.layout.RegularArray):
                        if maxsize > 1 and x.size == 1:
                            nextinputs.append(awkward1.layout.IndexedArray64(tmpindex, x.content).project())
                        elif x.size == maxsize:
                            nextinputs.append(x.content)
                        else:
                            raise ValueError("cannot broadcast RegularArray of size {0} with RegularArray of size {1}".format(x.size, maxsize))
                    else:
                        nextinputs.append(x)
                return awkward1.layout.RegularArray(apply(nextinputs), maxsize)

            else:
                for x in inputs:
                    if isinstance(x, awkward1._util.listtypes) and not isinstance(x, awkward1.layout.RegularArray):
                        first = x
                        break
                offsets = first.compact_offsets64()
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1._util.listtypes):
                        nextinputs.append(x.broadcast_tooffsets64(offsets).content)
                    # handle implicit left-broadcasting (unlike NumPy)
                    elif isinstance(x, awkward1.layout.Content):
                        nextinputs.append(awkward1.layout.RegularArray(x, 1).broadcast_tooffsets64(offsets).content)
                    else:
                        nextinputs.append(x)
                return awkward1.layout.ListOffsetArray64(offsets, apply(nextinputs))

        elif any(isinstance(x, awkward1._util.recordtypes) for x in inputs):
            keys = None
            length = None
            istuple = True
            for x in inputs:
                if isinstance(x, awkward1._util.recordtypes):
                    if keys is None:
                        keys = x.keys()
                    elif set(keys) != set(x.keys()):
                        raise ValueError("cannot broadcast records because keys don't match:\n    {0}\n    {1}".format(", ".join(sorted(keys)), ", ".join(sorted(x.keys()))))
                    if length is None:
                        length = len(x)
                    elif length != len(x):
                        raise ValueError("cannot broadcast RecordArray of length {0} with RecordArray of length {1}".format(length, len(x)))
                    if not x.istuple:
                        istuple = False

            if len(keys) == 0:
                return awkward1.layout.RecordArray(length, istuple)
            else:
                contents = []
                for key in keys:
                    contents.append(apply([x if not isinstance(x, awkward1._util.recordtypes) else x[key] for x in inputs]))
                return awkward1.layout.RecordArray(contents, keys)

        else:
            assert all(isinstance(x, awkward1.layout.NumpyArray) or not isinstance(x, awkward1.layout.Content) for x in inputs)
            result = getattr(ufunc, method)(*inputs, **kwargs)
            return awkward1.layout.NumpyArray(result)

    isscalar = []

    def pack(inputs):
        maxlen = -1
        for x in inputs:
            if isinstance(x, awkward1.layout.Content):
                maxlen = max(maxlen, len(x))
        if maxlen < 0:
            maxlen = 1
        nextinputs = []
        for x in inputs:
            if isinstance(x, awkward1.layout.Record):
                index = numpy.full(maxlen, x.at, dtype=numpy.int64)
                nextinputs.append(awkward1.layout.RegularArray(x.array[index], maxlen))
                isscalar.append(True)
            elif isinstance(x, awkward1.layout.Content):
                nextinputs.append(awkward1.layout.RegularArray(x, len(x)))
                isscalar.append(False)
            else:
                nextinputs.append(x)
                isscalar.append(True)
        return nextinputs

    def unpack(x):
        if all(isscalar):
            if len(x) == 0:
                return x.getitem_nothing().getitem_nothing()
            else:
                return x[0][0]
        else:
            if len(x) == 0:
                return x.getitem_nothing()
            else:
                return x[0]

    return awkward1._util.wrap(unpack(apply(pack([unwrap(x) for x in inputs]))), classes, functions)

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
