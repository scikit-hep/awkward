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
            raise TypeError("too many arguments for FillableArray.clear")

def call(context, builder, fcn, args, errormessage):
    fcntpe = context.get_function_pointer_type(fcn.numbatpe)
    fcnval = context.add_dynamic_addr(builder, fcn.numbatpe.get_pointer(fcn), info=fcn.name)
    fcnptr = builder.bitcast(fcnval, fcntpe)
    err = context.call_function_pointer(builder, fcnptr, args)
    with builder.if_then(builder.icmp_unsigned("!=", err, context.get_constant(numba.uint8, 0)), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, (errormessage,))

@numba.extending.lower_builtin("clear", FillableArrayType)
def lower_clear(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    call(context, builder, libawkward.FillableArray_clear, (proxyin.rawptr,), "could not clear FillableArray")
    return context.get_dummy_value()
