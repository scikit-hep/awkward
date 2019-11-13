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

    @numba.typing.templates.bound_function("integer")
    def resolve_integer(self, arraytpe, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], numba.types.Integer):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number of arguments for FillableArray.integer")

def call(context, builder, fcn, args):
    fcntpe = context.get_function_pointer_type(fcn.numbatpe)
    fcnval = context.add_dynamic_addr(builder, fcn.numbatpe.get_pointer(fcn), info=fcn.name)
    fcnptr = builder.bitcast(fcnval, fcntpe)
    err = context.call_function_pointer(builder, fcnptr, args)
    with builder.if_then(builder.icmp_unsigned("!=", err, context.get_constant(numba.types.uint8, 0)), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, (fcn.name + " failed",))

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

@numba.extending.lower_builtin("integer", FillableArrayType)
def lower_integer(context, builder, sig, args):
    tpe, xtpe = sig.args
    val, xval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    x = util.cast(context, builder, xtpe, numba.types.int64, xval)
    call(context, builder, libawkward.FillableArray_null, (proxyin.rawptr, x))
    return context.get_dummy_value()
