# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import operator

import numpy
import numba

import awkward1.operations.convert
import awkward1._util
import awkward1._numba.layout

class Lookup(object):
    def __init__(self, layout):
        positions = []
        arrays = []
        tolookup(layout, positions, arrays)

        def find(x):
            for i, array in enumerate(arrays):
                if x is array:
                    return i
            else:
                assert isinstance(x, int)
                return x

        self.positions = numpy.array([find(x) for x in positions], dtype=numpy.intp)
        self.arrays = tuple(arrays)
        self.arrayptrs = numpy.array([x.ctypes.data for x in arrays], dtype=numpy.intp)

        print("positions", self.positions)
        print("arrayptrs", self.arrayptrs)
        for i, x in enumerate(arrays):
            print("array[{0}] ".format(i), x)

def tolookup(layout, positions, arrays):
    import awkward1.layout

    if isinstance(layout, awkward1.layout.NumpyArray):
        return awkward1._numba.layout.NumpyArrayType.tolookup(layout, positions, arrays)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return awkward1._numba.layout.RegularArrayType.tolookup(layout, positions, arrays)

    elif isinstance(layout, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
        return awkward1._numba.layout.ListArrayType.tolookup(layout, positions, arrays)

    elif isinstance(layout, (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)):
        return awkward1._numba.layout.IndexedArrayType.tolookup(layout, positions, arrays)

    elif isinstance(layout, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
        return awkward1._numba.layout.IndexedOptionArrayType.tolookup(layout, positions, arrays)

    elif isinstance(layout, awkward1.layout.RecordArray):
        return awkward1._numba.layout.RecordArrayType.tolookup(layout, positions, arrays)

    elif isinstance(layout, awkward1.layout.Record):
        return awkward1._numba.layout.RecordType.tolookup(layout, positions, arrays)

    elif isinstance(layout, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
        return awkward1._numba.layout.UnionArrayType.tolookup(layout, positions, arrays)

    else:
        raise AssertionError("unrecognized layout type: {0}".format(type(layout)))

@numba.extending.typeof_impl.register(Lookup)
def typeof_Lookup(obj, c):
    return LookupType()

class LookupType(numba.types.Type):
    arraytype = numba.types.Array(numba.intp, 1, "C")

    def __init__(self):
        super(LookupType, self).__init__(name="LookupType()")

@numba.extending.register_model(LookupType)
class LookupModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("positions", fe_type.arraytype),
                   ("arrayptrs", fe_type.arraytype)]
        super(LookupModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(LookupType)
def unbox_Lookup(lookuptype, lookupobj, c):
    positions_obj = c.pyapi.object_getattr_string(lookupobj, "positions")
    arrayptrs_obj = c.pyapi.object_getattr_string(lookupobj, "arrayptrs")

    proxyout = c.context.make_helper(c.builder, lookuptype)
    proxyout.positions = c.pyapi.to_native_value(lookuptype.arraytype, positions_obj).value
    proxyout.arrayptrs = c.pyapi.to_native_value(lookuptype.arraytype, arrayptrs_obj).value

    c.pyapi.decref(positions_obj)
    c.pyapi.decref(arrayptrs_obj)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

class ArrayView(object):
    @classmethod
    def fromarray(self, array):
        behavior = awkward1._util.behaviorof(array)
        layout = awkward1.operations.convert.tolayout(array, allowrecord=True, allowother=False, numpytype=(numpy.number,))
        layout = awkward1.operations.convert.regularize_numpyarray(layout, allowempty=False, highlevel=False)
        return ArrayView(numba.typeof(layout), behavior, Lookup(layout), 0, 0, len(layout), ())

    def __init__(self, type, behavior, lookup, pos, start, stop, fields):
        self.type = type
        self.behavior = behavior
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields

    def toarray(self):
        layout = self.type.tolayout(self.lookup, self.pos, self.fields)
        return awkward1._util.wrap(layout[self.start:self.stop], self.behavior)

@numba.extending.typeof_impl.register(ArrayView)
def typeof_ArrayView(obj, c):
    return ArrayViewType(obj.type, obj.behavior, obj.fields)

class ArrayViewType(numba.types.Type):
    def __init__(self, type, behavior, fields):
        super(ArrayViewType, self).__init__(name="awkward1.ArrayView({0}, {1}, {2})".format(type.name, "None" if behavior is None else "<Behavior {0}>".format(hash(behavior)), repr(fields)))
        self.type = type
        self.behavior = behavior
        self.fields = fields

@numba.extending.register_model(ArrayViewType)
class ArrayViewModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("pos",       numba.intp),
                   ("start",     numba.intp),
                   ("stop",      numba.intp),
                   ("positions", numba.types.CPointer(numba.intp)),
                   ("arrayptrs", numba.types.CPointer(numba.intp)),
                   ("pylookup",  numba.types.pyobject)]
        super(ArrayViewModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ArrayViewType)
def unbox_ArrayView(viewtype, arrayobj, c):
    view_obj   = c.pyapi.object_getattr_string(arrayobj, "_numbaview")
    lookup_obj = c.pyapi.object_getattr_string(view_obj, "lookup")
    pos_obj    = c.pyapi.object_getattr_string(view_obj, "pos")
    start_obj  = c.pyapi.object_getattr_string(view_obj, "start")
    stop_obj   = c.pyapi.object_getattr_string(view_obj, "stop")

    lookup_val = c.pyapi.to_native_value(LookupType(), lookup_obj).value
    lookup_proxy = c.context.make_helper(c.builder, LookupType(), lookup_val)

    proxyout = c.context.make_helper(c.builder, viewtype)
    proxyout.pos       = c.pyapi.number_as_ssize_t(pos_obj)
    proxyout.start     = c.pyapi.number_as_ssize_t(start_obj)
    proxyout.stop      = c.pyapi.number_as_ssize_t(stop_obj)
    proxyout.positions = c.context.make_helper(c.builder, LookupType.arraytype, lookup_proxy.positions).data
    proxyout.arrayptrs = c.context.make_helper(c.builder, LookupType.arraytype, lookup_proxy.arrayptrs).data
    proxyout.pylookup  = lookup_obj

    c.pyapi.decref(view_obj)
    c.pyapi.decref(lookup_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder, LookupType(), lookup_val)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ArrayViewType)
def box_ArrayView(viewtype, viewval, c):
    ArrayView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ArrayView))
    type_obj      = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.type))
    behavior_obj  = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.behavior))
    fields_obj    = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.fields))

    proxyin = c.context.make_helper(c.builder, viewtype, viewval)
    pos_obj    = c.pyapi.long_from_ssize_t(proxyin.pos)
    start_obj  = c.pyapi.long_from_ssize_t(proxyin.start)
    stop_obj   = c.pyapi.long_from_ssize_t(proxyin.stop)
    lookup_obj = proxyin.pylookup

    arrayview_obj = c.pyapi.call_function_objargs(ArrayView_obj, (type_obj, behavior_obj, lookup_obj, pos_obj, start_obj, stop_obj, fields_obj))

    out = c.pyapi.call_method(arrayview_obj, "toarray", ())

    c.pyapi.decref(ArrayView_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(fields_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)
    c.pyapi.decref(arrayview_obj)

    return out

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            return numba.intp(args[0])

@numba.extending.lower_builtin(len, ArrayViewType)
def lower_len(context, builder, sig, args):
    proxyin = context.make_helper(builder, sig.args[0], args[0])
    return builder.sub(proxyin.stop, proxyin.start)

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            viewtype, wheretype = args
            if isinstance(wheretype, numba.types.Integer):
                return viewtype.type.getitem_at(viewtype)(viewtype, wheretype)
            elif isinstance(wheretype, numba.types.SliceType) and not wheretype.has_step:
                return viewtype.type.getitem_range(viewtype)(viewtype, wheretype)
            elif isinstance(wheretype, numba.types.StringLiteral):
                return viewtype.type.getitem_field(viewtype, wheretype.literal_value)(viewtype, wheretype)
            else:
                raise TypeError("only an integer, start:stop range, or a field name string may be used as Awkward Array slices in compiled code")

@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.Integer)
def lower_getitem_at(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    return viewtype.type.lower_getitem_at(context, builder, rettype, viewtype, viewval, viewproxy, wheretype, whereval, True, True)

@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    whereproxy = context.make_helper(builder, wheretype, whereval)
    return viewtype.type.lower_getitem_range(context, builder, rettype, viewtype, viewval, viewproxy, whereproxy.start, whereproxy.stop, True)

@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.StringLiteral)
def lower_getitem_field(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    return viewtype.type.lower_getitem_field(context, builder, rettype, viewtype, viewval, viewproxy, wheretype.literal_value)

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = ArrayViewType

    def generic_resolve(self, viewtype, attr):
        # if attr == "???":
        #     do_something_specific
        return viewtype.getitem_field(attr)

@numba.extending.lower_getattr_generic(ArrayViewType)
def lower_getattr_generic(context, builder, viewtype, viewval, attr):
    viewproxy = context.make_helper(builder, viewtype, viewval)
    return viewtype.lower_getitem_field(context, builder, viewtype, viewval, viewproxy, attr)
