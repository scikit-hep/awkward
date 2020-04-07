# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import operator

import numpy
import numba

import awkward1.operations.convert
import awkward1._util
import awkward1._connect._numba.layout

class Lookup(object):
    def __init__(self, layout):
        positions = []
        sharedptrs = []
        arrays = []
        tolookup(layout, positions, sharedptrs, arrays)
        assert len(positions) == len(sharedptrs)

        def find(x):
            for i, array in enumerate(arrays):
                if x is array:
                    return i
            else:
                assert isinstance(x, int)
                return x

        self.positions = [find(x) for x in positions]
        self.sharedptrs_hold = sharedptrs
        self.arrays = tuple(arrays)
        self.arrayptrs = numpy.array([x if isinstance(x, int)
                                        else x.ctypes.data for x in positions],
                                     dtype=numpy.intp)
        self.sharedptrs = numpy.array([0 if x is None else x.ptr()
                                         for x in sharedptrs],
                                      dtype=numpy.intp)

def tolookup(layout, positions, sharedptrs, arrays):
    import awkward1.layout

    if isinstance(layout, awkward1.layout.NumpyArray):
        return awkward1._connect._numba.layout.NumpyArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return awkward1._connect._numba.layout.RegularArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, (awkward1.layout.ListArray32,
                             awkward1.layout.ListArrayU32,
                             awkward1.layout.ListArray64,
                             awkward1.layout.ListOffsetArray32,
                             awkward1.layout.ListOffsetArrayU32,
                             awkward1.layout.ListOffsetArray64)):
        return awkward1._connect._numba.layout.ListArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, (awkward1.layout.IndexedArray32,
                             awkward1.layout.IndexedArrayU32,
                             awkward1.layout.IndexedArray64)):
        return awkward1._connect._numba.layout.IndexedArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, (awkward1.layout.IndexedOptionArray32,
                             awkward1.layout.IndexedOptionArray64)):
        return awkward1._connect._numba.layout.IndexedOptionArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, awkward1.layout.ByteMaskedArray):
        return awkward1._connect._numba.layout.ByteMaskedArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, awkward1.layout.BitMaskedArray):
        return awkward1._connect._numba.layout.BitMaskedArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, awkward1.layout.UnmaskedArray):
        return awkward1._connect._numba.layout.UnmaskedArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, awkward1.layout.RecordArray):
        return awkward1._connect._numba.layout.RecordArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, awkward1.layout.Record):
        return awkward1._connect._numba.layout.RecordType.tolookup(
                 layout, positions, sharedptrs, arrays)

    elif isinstance(layout, (awkward1.layout.UnionArray8_32,
                             awkward1.layout.UnionArray8_U32,
                             awkward1.layout.UnionArray8_64)):
        return awkward1._connect._numba.layout.UnionArrayType.tolookup(
                 layout, positions, sharedptrs, arrays)

    else:
        raise AssertionError(
                "unrecognized layout type: {0}".format(type(layout)))

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
        members = [("arrayptrs", fe_type.arraytype),
                   ("sharedptrs", fe_type.arraytype)]
        super(LookupModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(LookupType)
def unbox_Lookup(lookuptype, lookupobj, c):
    arrayptrs_obj = c.pyapi.object_getattr_string(lookupobj, "arrayptrs")
    sharedptrs_obj = c.pyapi.object_getattr_string(lookupobj, "sharedptrs")

    proxyout = c.context.make_helper(c.builder, lookuptype)
    proxyout.arrayptrs = c.pyapi.to_native_value(lookuptype.arraytype,
                                                 arrayptrs_obj).value
    proxyout.sharedptrs = c.pyapi.to_native_value(lookuptype.arraytype,
                                                  sharedptrs_obj).value

    c.pyapi.decref(arrayptrs_obj)
    c.pyapi.decref(sharedptrs_obj)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

class ArrayView(object):
    @classmethod
    def fromarray(self, array):
        behavior = awkward1._util.behaviorof(array)
        layout = awkward1.operations.convert.to_layout(
                   array,
                   allow_record=False,
                   allow_other=False,
                   numpytype=(numpy.number, numpy.bool_, numpy.bool))
        layout = awkward1.operations.convert.regularize_numpyarray(
                   layout,
                   allow_empty=False,
                   highlevel=False)
        return ArrayView(numba.typeof(layout),
                         behavior,
                         Lookup(layout),
                         0,
                         0,
                         len(layout),
                         ())

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

def wrap(type, viewtype, fields):
    if fields is None:
        return ArrayViewType(type, viewtype.behavior, viewtype.fields)
    else:
        return ArrayViewType(type, viewtype.behavior, fields)

class ArrayViewType(numba.types.Type):
    def __init__(self, type, behavior, fields):
        super(ArrayViewType, self).__init__(
            name="awkward1.ArrayView({0}, {1}, {2})".format(
              type.name,
              awkward1._connect._numba.repr_behavior(behavior),
              repr(fields)))
        self.type = type
        self.behavior = behavior
        self.fields = fields

@numba.extending.register_model(ArrayViewType)
class ArrayViewModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("pos",        numba.intp),
                   ("start",      numba.intp),
                   ("stop",       numba.intp),
                   ("arrayptrs",  numba.types.CPointer(numba.intp)),
                   ("sharedptrs", numba.types.CPointer(numba.intp)),
                   ("pylookup",   numba.types.pyobject)]
        super(ArrayViewModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ArrayViewType)
def unbox_Array(viewtype, arrayobj, c):
    view_obj   = c.pyapi.object_getattr_string(arrayobj, "_numbaview")
    out = unbox_ArrayView(viewtype, view_obj, c)
    c.pyapi.decref(view_obj)
    return out

def unbox_ArrayView(viewtype, view_obj, c):
    lookup_obj = c.pyapi.object_getattr_string(view_obj, "lookup")
    pos_obj    = c.pyapi.object_getattr_string(view_obj, "pos")
    start_obj  = c.pyapi.object_getattr_string(view_obj, "start")
    stop_obj   = c.pyapi.object_getattr_string(view_obj, "stop")

    lookup_val = c.pyapi.to_native_value(LookupType(), lookup_obj).value
    lookup_proxy = c.context.make_helper(c.builder, LookupType(), lookup_val)

    proxyout = c.context.make_helper(c.builder, viewtype)
    proxyout.pos        = c.pyapi.number_as_ssize_t(pos_obj)
    proxyout.start      = c.pyapi.number_as_ssize_t(start_obj)
    proxyout.stop       = c.pyapi.number_as_ssize_t(stop_obj)
    proxyout.arrayptrs  = c.context.make_helper(c.builder,
                                                LookupType.arraytype,
                                                lookup_proxy.arrayptrs).data
    proxyout.sharedptrs = c.context.make_helper(c.builder,
                                                LookupType.arraytype,
                                                lookup_proxy.sharedptrs).data
    proxyout.pylookup   = lookup_obj

    c.pyapi.decref(lookup_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder, LookupType(), lookup_val)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ArrayViewType)
def box_Array(viewtype, viewval, c):
    arrayview_obj = box_ArrayView(viewtype, viewval, c)
    out = c.pyapi.call_method(arrayview_obj, "toarray", ())
    c.pyapi.decref(arrayview_obj)
    return out

def dict2serializable(obj):
    if obj is None:
        return None
    else:
        return tuple(obj.items())

def serializable2dict(obj):
    if obj is None:
        return None
    else:
        return dict(obj)

def box_ArrayView(viewtype, viewval, c):
    serializable2dict_obj = c.pyapi.unserialize(
                              c.pyapi.serialize_object(serializable2dict))
    behavior2_obj = c.pyapi.unserialize(
                      c.pyapi.serialize_object(
                        dict2serializable(viewtype.behavior)))
    behavior_obj  = c.pyapi.call_function_objargs(serializable2dict_obj,
                                                  (behavior2_obj,))
    ArrayView_obj = c.pyapi.unserialize(
                      c.pyapi.serialize_object(ArrayView))
    type_obj      = c.pyapi.unserialize(
                      c.pyapi.serialize_object(viewtype.type))
    fields_obj    = c.pyapi.unserialize(
                      c.pyapi.serialize_object(viewtype.fields))

    proxyin = c.context.make_helper(c.builder, viewtype, viewval)
    pos_obj    = c.pyapi.long_from_ssize_t(proxyin.pos)
    start_obj  = c.pyapi.long_from_ssize_t(proxyin.start)
    stop_obj   = c.pyapi.long_from_ssize_t(proxyin.stop)
    lookup_obj = proxyin.pylookup

    out = c.pyapi.call_function_objargs(ArrayView_obj,
                                        (type_obj,
                                         behavior_obj,
                                         lookup_obj,
                                         pos_obj,
                                         start_obj,
                                         stop_obj,
                                         fields_obj))

    c.pyapi.decref(serializable2dict_obj)
    c.pyapi.decref(behavior2_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(ArrayView_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(fields_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    return out

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if (len(args) == 1 and
            len(kwargs) == 0 and isinstance(args[0], ArrayViewType)):
            return numba.intp(args[0])

@numba.extending.lower_builtin(len, ArrayViewType)
def lower_len(context, builder, sig, args):
    proxyin = context.make_helper(builder, sig.args[0], args[0])
    return builder.sub(proxyin.stop, proxyin.start)

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if (len(args) == 2 and
            len(kwargs) == 0 and
            isinstance(args[0], ArrayViewType)):
            viewtype, wheretype = args
            if isinstance(wheretype, numba.types.Integer):
                return viewtype.type.getitem_at_check(viewtype)(viewtype,
                                                                wheretype)
            elif (isinstance(wheretype, numba.types.SliceType) and
                  not wheretype.has_step):
                return viewtype.type.getitem_range(viewtype)(viewtype,
                                                             wheretype)
            elif isinstance(wheretype, numba.types.StringLiteral):
                return viewtype.type.getitem_field(
                         viewtype, wheretype.literal_value)(viewtype,
                                                            wheretype)
            else:
                raise TypeError(
                        "only an integer, start:stop range, or a *constant* "
                        "field name string may be used as awkward1.Array "
                        "slices in compiled code")

@numba.extending.lower_builtin(operator.getitem,
                               ArrayViewType,
                               numba.types.Integer)
def lower_getitem_at(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    return viewtype.type.lower_getitem_at_check(context,
                                                builder,
                                                rettype,
                                                viewtype,
                                                viewval,
                                                viewproxy,
                                                wheretype,
                                                whereval,
                                                True,
                                                True)

@numba.extending.lower_builtin(operator.getitem,
                               ArrayViewType,
                               numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    whereproxy = context.make_helper(builder, wheretype, whereval)
    return viewtype.type.lower_getitem_range(context,
                                             builder,
                                             rettype,
                                             viewtype,
                                             viewval,
                                             viewproxy,
                                             whereproxy.start,
                                             whereproxy.stop,
                                             True)

@numba.extending.lower_builtin(operator.getitem,
                               ArrayViewType,
                               numba.types.StringLiteral)
def lower_getitem_field(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    return viewtype.type.lower_getitem_field(context,
                                             builder,
                                             viewtype,
                                             viewval,
                                             wheretype.literal_value)

@numba.typing.templates.infer_getattr
class type_getattr(numba.typing.templates.AttributeTemplate):
    key = ArrayViewType

    def generic_resolve(self, viewtype, attr):
        # if attr == "???":
        #     do_something_specific
        return viewtype.type.getitem_field(viewtype, attr)

@numba.extending.lower_getattr_generic(ArrayViewType)
def lower_getattr_generic(context, builder, viewtype, viewval, attr):
    return viewtype.type.lower_getitem_field(context,
                                             builder,
                                             viewtype,
                                             viewval,
                                             attr)

class IteratorType(numba.types.common.SimpleIteratorType):
    def __init__(self, viewtype):
        super(IteratorType, self).__init__("awkward1.Iterator({0})".format(
                viewtype.name), viewtype.type.getitem_at_check(viewtype))
        self.viewtype = viewtype

@numba.typing.templates.infer
class type_getiter(numba.typing.templates.AbstractTemplate):
    key = "getiter"

    def generic(self, args, kwargs):
        if (len(args) == 1 and
            len(kwargs) == 0 and
            isinstance(args[0], ArrayViewType)):
            return IteratorType(args[0])(args[0])

@numba.datamodel.registry.register_default(IteratorType)
class IteratorModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("view", fe_type.viewtype),
                   ("length", numba.intp),
                   ("at", numba.types.EphemeralPointer(numba.intp))]
        super(IteratorModel, self).__init__(dmm, fe_type, members)

@numba.extending.lower_builtin("getiter", ArrayViewType)
def lower_getiter(context, builder, sig, args):
    rettype, (viewtype,) = sig.return_type, sig.args
    viewval, = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    proxyout = context.make_helper(builder, rettype)
    proxyout.view = viewval
    proxyout.length = builder.sub(viewproxy.stop, viewproxy.start)
    proxyout.at = numba.cgutils.alloca_once_value(
                    builder, context.get_constant(numba.intp, 0))
    if context.enable_nrt:
        context.nrt.incref(builder, viewtype, viewval)
    return numba.targets.imputils.impl_ret_new_ref(context,
                                                   builder,
                                                   rettype,
                                                   proxyout._getvalue())

@numba.extending.lower_builtin("iternext", IteratorType)
@numba.targets.imputils.iternext_impl(numba.targets.imputils.RefType.BORROWED)
def lower_iternext(context, builder, sig, args, result):
    itertype, = sig.args
    iterval, = args
    proxyin = context.make_helper(builder, itertype, iterval)
    at = builder.load(proxyin.at)

    is_valid = builder.icmp_signed("<", at, proxyin.length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        result.yield_(lower_getitem_at(
            context,
            builder,
            itertype.yield_type(itertype.viewtype, numba.intp),
            (proxyin.view, at)))
        nextat = numba.cgutils.increment_index(builder, at)
        builder.store(nextat, proxyin.at)

class RecordView(object):
    @classmethod
    def fromrecord(self, record):
        behavior = awkward1._util.behaviorof(record)
        layout = awkward1.operations.convert.to_layout(
                   record,
                   allow_record=True,
                   allow_other=False,
                   numpytype=(numpy.number, numpy.bool_, numpy.bool))
        assert isinstance(layout, awkward1.layout.Record)
        arraylayout = layout.array
        return RecordView(ArrayView(numba.typeof(arraylayout),
                                    behavior,
                                    Lookup(arraylayout),
                                    0,
                                    0,
                                    len(arraylayout),
                                    ()),
                          layout.at)

    def __init__(self, arrayview, at):
        self.arrayview = arrayview
        self.at = at

    def torecord(self):
        arraylayout = self.arrayview.toarray().layout
        return awkward1._util.wrap(
                 awkward1.layout.Record(arraylayout, self.at),
                 self.arrayview.behavior)

@numba.extending.typeof_impl.register(RecordView)
def typeof_RecordView(obj, c):
    return RecordViewType(numba.typeof(obj.arrayview))

class RecordViewType(numba.types.Type):
    def __init__(self, arrayviewtype):
        super(RecordViewType, self).__init__(
            name="RecordViewType({0})".format(arrayviewtype.name))
        self.arrayviewtype = arrayviewtype

    @property
    def behavior(self):
        return self.arrayviewtype.behavior

    @property
    def fields(self):
        return self.arrayviewtype.fields

    def typer_field(self, key):
        return self.arrayviewtype.type.getitem_field_record(self, key)

    def lower_field(self, context, builder, val, key):
        return self.arrayviewtype.type.lower_getitem_field_record(context,
                                                                  builder,
                                                                  self,
                                                                  val,
                                                                  key)

@numba.extending.register_model(RecordViewType)
class RecordViewModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("arrayview", fe_type.arrayviewtype),
                   ("at",        numba.intp)]
        super(RecordViewModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(RecordViewType)
def unbox_RecordView(recordviewtype, recordobj, c):
    recordview_obj = c.pyapi.object_getattr_string(recordobj, "_numbaview")
    arrayview_obj  = c.pyapi.object_getattr_string(recordview_obj, "arrayview")
    at_obj         = c.pyapi.object_getattr_string(recordview_obj, "at")

    arrayview_val = unbox_ArrayView(recordviewtype.arrayviewtype,
                                    arrayview_obj,
                                    c).value

    proxyout = c.context.make_helper(c.builder, recordviewtype)
    proxyout.arrayview = arrayview_val
    proxyout.at        = c.pyapi.number_as_ssize_t(at_obj)

    c.pyapi.decref(recordview_obj)
    c.pyapi.decref(at_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder,
                             recordviewtype.arrayviewtype,
                             arrayview_val)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(RecordViewType)
def box_RecordView(recordviewtype, viewval, c):
    RecordView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(RecordView))

    proxyin = c.context.make_helper(c.builder, recordviewtype, viewval)
    arrayview_obj = box_ArrayView(recordviewtype.arrayviewtype,
                                  proxyin.arrayview,
                                  c)
    at_obj        = c.pyapi.long_from_ssize_t(proxyin.at)

    recordview_obj = c.pyapi.call_function_objargs(RecordView_obj,
                                                   (arrayview_obj, at_obj))

    out = c.pyapi.call_method(recordview_obj, "torecord", ())

    c.pyapi.decref(RecordView_obj)
    c.pyapi.decref(arrayview_obj)
    c.pyapi.decref(at_obj)
    c.pyapi.decref(recordview_obj)

    return out

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem_record(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if (len(args) == 2 and
            len(kwargs) == 0 and
            isinstance(args[0], RecordViewType)):
            recordviewtype, wheretype = args
            if isinstance(wheretype, numba.types.StringLiteral):
                return recordviewtype.arrayviewtype.type.getitem_field_record(
                         recordviewtype,
                         wheretype.literal_value)(recordviewtype, wheretype)
            else:
                raise TypeError(
                        "only a *constant* field name string may be used as "
                        "awkward1.Record slices in compiled code")

@numba.extending.lower_builtin(operator.getitem,
                               RecordViewType,
                               numba.types.StringLiteral)
def lower_getitem_field_record(context, builder, sig, args):
    rettype, (recordviewtype, wheretype) = sig.return_type, sig.args
    recordviewval, whereval = args
    return recordviewtype.arrayviewtype.type.lower_getitem_field_record(
             context,
             builder,
             recordviewtype,
             recordviewval,
             wheretype.literal_value)

@numba.typing.templates.infer_getattr
class type_getattr_record(numba.typing.templates.AttributeTemplate):
    key = RecordViewType

    def generic_resolve(self, recordviewtype, attr):
        for methodname, typer, lower in awkward1._util.numba_methods(
                                        recordviewtype.arrayviewtype.type,
                                        recordviewtype.arrayviewtype.behavior):
            if attr == methodname:
                class type_method(numba.typing.templates.AbstractTemplate):
                    key = methodname
                    def generic(_, args, kwargs):
                        if len(kwargs) == 0:
                            sig = typer(recordviewtype, args)
                            sig.recvr = recordviewtype
                            numba.extending.lower_builtin(
                                methodname,
                                recordviewtype,
                                *[x.literal_type
                                    if isinstance(x, numba.types.Literal)
                                    else x for x in args])(lower)
                            return sig
                return numba.types.BoundFunction(type_method, recordviewtype)

        for attrname, typer, lower in awkward1._util.numba_attrs(
                                      recordviewtype.arrayviewtype.type,
                                      recordviewtype.arrayviewtype.behavior):
            if attr == attrname:
                return typer(recordviewtype)

        else:
            return recordviewtype.typer_field(attr)

@numba.extending.lower_getattr_generic(RecordViewType)
def lower_getattr_generic_record(context,
                                 builder,
                                 recordviewtype,
                                 recordviewval,
                                 attr):
    for attrname, typer, lower in awkward1._util.numba_attrs(
                                  recordviewtype.arrayviewtype.type,
                                  recordviewtype.arrayviewtype.behavior):
        if attr == attrname:
            return lower(context,
                         builder,
                         typer(recordviewtype)(recordviewtype),
                         (recordviewval,))
    else:
        return recordviewtype.lower_field(context,
                                          builder,
                                          recordviewval,
                                          attr)

def register_unary_operator(unaryop):
    @numba.typing.templates.infer_global(unaryop)
    class type_binary_operator(numba.typing.templates.AbstractTemplate):
        def generic(self, args, kwargs):
            if len(args) == 1 and len(kwargs) == 0:
                behavior = None

                if isinstance(args[0], RecordViewType):
                    left = args[0].arrayviewtype.type
                    behavior = args[0].arrayviewtype.behavior

                for typer, lower in awkward1._util.numba_unaryops(unaryop,
                                                                  left,
                                                                  behavior):
                    numba.extending.lower_builtin(unaryop, *args)(lower)
                    return typer(unaryop, args[0])

for unaryop in (abs,
                operator.inv,
                operator.invert,
                operator.neg,
                operator.not_,
                operator.pos,
                operator.truth):
    register_unary_operator(unaryop)

def register_binary_operator(binop):
    @numba.typing.templates.infer_global(binop)
    class type_binary_operator(numba.typing.templates.AbstractTemplate):
        def generic(self, args, kwargs):
            if len(args) == 2 and len(kwargs) == 0:
                behavior = None

                if isinstance(args[0], RecordViewType):
                    left = args[0].arrayviewtype.type
                    behavior = args[0].arrayviewtype.behavior

                if isinstance(args[1], RecordViewType):
                    right = args[1].arrayviewtype.type
                    if behavior is None:
                        behavior = args[1].arrayviewtype.behavior

                for typer, lower in awkward1._util.numba_binops(binop,
                                                                left,
                                                                right,
                                                                behavior):
                    numba.extending.lower_builtin(binop, *args)(lower)
                    return typer(binop, args[0], args[1])

for binop in ((operator.add,
               operator.and_,
               operator.contains,
               operator.eq,
               operator.floordiv,
               operator.ge,
               operator.gt,
               operator.le,
               operator.lshift,
               operator.lt,
               operator.mod,
               operator.mul,
               operator.ne,
               operator.or_,
               operator.pow,
               operator.rshift,
               operator.sub,
               operator.truediv,
               operator.xor)
              + (() if not hasattr(operator, "matmul")
                    else (operator.matmul,))):
    register_binary_operator(binop)
