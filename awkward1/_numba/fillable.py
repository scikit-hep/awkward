# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numba

import awkward1.layout
from .._numba import libawkward, util

@numba.extending.typeof_impl.register(awkward1.layout.FillableArray)
def typeof(val, c):
    return FillableArrayType()

class FillableArrayType(numba.types.Type):
    def __init__(self):
        super(FillableArrayType, self).__init__("FillableArrayType")

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytpe, = args
            if isinstance(arraytpe, FillableArrayType):
                return numba.typing.templates.signature(numba.types.intp, arraytpe)

@numba.datamodel.registry.register_default(FillableArrayType)
class FillableArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("rawptr", numba.types.voidptr),
                   ("pyptr", numba.types.pyobject)]
        super(FillableArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(FillableArrayType)
def unbox(tpe, obj, c):
    rawptr_obj = c.pyapi.object_getattr_string(obj, "_ptr")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.rawptr = c.pyapi.long_as_voidptr(rawptr_obj)
    proxyout.pyptr = obj
    c.pyapi.decref(rawptr_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(FillableArrayType)
def box(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    c.pyapi.incref(proxyin.pyptr)
    return proxyin.pyptr

def call(context, builder, fcn, args):
    fcntpe = context.get_function_pointer_type(fcn.numbatpe)
    fcnval = context.add_dynamic_addr(builder, fcn.numbatpe.get_pointer(fcn), info=fcn.name)
    fcnptr = builder.bitcast(fcnval, fcntpe)
    err = context.call_function_pointer(builder, fcnptr, args)
    with builder.if_then(builder.icmp_unsigned("!=", err, context.get_constant(numba.uint8, 0)), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, (fcn.name + " failed",))

@numba.extending.lower_builtin(len, FillableArrayType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    result = numba.cgutils.alloca_once(builder, context.get_value_type(numba.int64))
    call(context, builder, libawkward.FillableArray_length, (proxyin.rawptr, result))
    return util.cast(context, builder, numba.int64, numba.intp, builder.load(result))

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = FillableArrayType

    @numba.typing.templates.bound_function("clear")
    def resolve_clear(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for FillableArray.clear")

    @numba.typing.templates.bound_function("null")
    def resolve_null(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for FillableArray.null")

    @numba.typing.templates.bound_function("boolean")
    def resolve_boolean(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.Boolean):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number or types of arguments for FillableArray.boolean")

    @numba.typing.templates.bound_function("integer")
    def resolve_integer(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.Integer):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number or types of arguments for FillableArray.integer")

    @numba.typing.templates.bound_function("real")
    def resolve_real(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (numba.types.Integer, numba.types.Float)):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number or types of arguments for FillableArray.real")

    @numba.typing.templates.bound_function("beginlist")
    def resolve_beginlist(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for FillableArray.beginlist")

    @numba.typing.templates.bound_function("endlist")
    def resolve_endlist(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for FillableArray.endlist")

    @numba.typing.templates.bound_function("begintuple")
    def resolve_begintuple(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.Integer):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number of arguments for FillableArray.begintuple")

    @numba.typing.templates.bound_function("index")
    def resolve_index(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.Integer):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number of arguments for FillableArray.index")

    @numba.typing.templates.bound_function("endtuple")
    def resolve_endtuple(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for FillableArray.endtuple")

    @numba.typing.templates.bound_function("beginrecord")
    def resolve_beginrecord(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        elif len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.StringLiteral):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number of arguments for FillableArray.beginrecord")

    @numba.typing.templates.bound_function("field")
    def resolve_field(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.StringLiteral):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number of arguments for FillableArray.field")

    @numba.typing.templates.bound_function("endrecord")
    def resolve_endrecord(self, arraytpe, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for FillableArray.endrecord")

@numba.extending.lower_builtin("clear", FillableArrayType)
def lower_clear(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_clear, (proxyin.rawptr,))
    return context.get_dummy_value()

@numba.extending.lower_builtin("null", FillableArrayType)
def lower_null(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_null, (proxyin.rawptr,))
    return context.get_dummy_value()

@numba.extending.lower_builtin("boolean", FillableArrayType, numba.types.Boolean)
def lower_integer(context, builder, sig, args):
    tpe, xtpe = sig.args
    val, xval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    x = builder.zext(xval, context.get_value_type(numba.uint8))
    call(context, builder, libawkward.FillableArray_boolean, (proxyin.rawptr, x))
    return context.get_dummy_value()

@numba.extending.lower_builtin("integer", FillableArrayType, numba.types.Integer)
def lower_integer(context, builder, sig, args):
    tpe, xtpe = sig.args
    val, xval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    x = util.cast(context, builder, xtpe, numba.int64, xval)
    call(context, builder, libawkward.FillableArray_integer, (proxyin.rawptr, x))
    return context.get_dummy_value()

@numba.extending.lower_builtin("real", FillableArrayType, numba.types.Integer)
@numba.extending.lower_builtin("real", FillableArrayType, numba.types.Float)
def lower_real(context, builder, sig, args):
    tpe, xtpe = sig.args
    val, xval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if isinstance(xtpe, numba.types.Integer) and xtpe.signed:
        x = builder.sitofp(xval, context.get_value_type(numba.types.float64))
    elif isinstance(xtpe, numba.types.Integer):
        x = builder.uitofp(xval, context.get_value_type(numba.types.float64))
    elif xtpe.bitwidth < 64:
        x = builder.fpext(xval, context.get_value_type(numba.types.float64))
    elif xtpe.bitwidth > 64:
        x = builder.fptrunc(xval, context.get_value_type(numba.types.float64))
    else:
        x = xval
    call(context, builder, libawkward.FillableArray_real, (proxyin.rawptr, x))
    return context.get_dummy_value()

@numba.extending.lower_builtin("beginlist", FillableArrayType)
def lower_beginlist(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_beginlist, (proxyin.rawptr,))
    return context.get_dummy_value()

@numba.extending.lower_builtin("endlist", FillableArrayType)
def lower_endlist(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_endlist, (proxyin.rawptr,))
    return context.get_dummy_value()

@numba.extending.lower_builtin("begintuple", FillableArrayType, numba.types.Integer)
def lower_begintuple(context, builder, sig, args):
    tpe, numfieldstpe = sig.args
    val, numfieldsval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    numfields = util.cast(context, builder, numfieldstpe, numba.int64, numfieldsval)
    call(context, builder, libawkward.FillableArray_begintuple, (proxyin.rawptr, numfields))
    return context.get_dummy_value()

@numba.extending.lower_builtin("index", FillableArrayType, numba.types.Integer)
def lower_index(context, builder, sig, args):
    tpe, indextpe = sig.args
    val, indexval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    index = util.cast(context, builder, indextpe, numba.int64, indexval)
    call(context, builder, libawkward.FillableArray_index, (proxyin.rawptr, index))
    return context.get_dummy_value()

@numba.extending.lower_builtin("endtuple", FillableArrayType)
def lower_endtuple(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_endtuple, (proxyin.rawptr,))
    return context.get_dummy_value()

@numba.extending.lower_builtin("beginrecord", FillableArrayType)
def lower_beginrecord(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_beginrecord, (proxyin.rawptr, context.get_constant(numba.int64, 0)))
    return context.get_dummy_value()

@numba.extending.lower_builtin("beginrecord", FillableArrayType, numba.types.StringLiteral)
def lower_beginrecord(context, builder, sig, args):
    tpe, nametpe = sig.args
    val, nameval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    name = util.globalstring(context, builder, nametpe.literal_value, inttype=numba.int64)
    call(context, builder, libawkward.FillableArray_beginrecord, (proxyin.rawptr, name))
    return context.get_dummy_value()

@numba.extending.lower_builtin("field", FillableArrayType, numba.types.StringLiteral)
def lower_field(context, builder, sig, args):
    tpe, keytpe = sig.args
    val, keyval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    key = util.globalstring(context, builder, keytpe.literal_value)
    call(context, builder, libawkward.FillableArray_field_fast, (proxyin.rawptr, key))
    return context.get_dummy_value()

@numba.extending.lower_builtin("endrecord", FillableArrayType)
def lower_endrecord(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_endrecord, (proxyin.rawptr,))
    return context.get_dummy_value()
