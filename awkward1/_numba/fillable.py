# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import numba

import awkward1.operations.convert
import awkward1._util
import awkward1._numba.layout
import awkward1._numba.libawkward

class FillableArrayType(numba.types.Type):
    def __init__(self, behavior):
        super(FillableArrayType, self).__init__(name="awkward1.FillableArrayType({0})".format(awkward1._numba.repr_behavior(behavior)))
        self.behavior = behavior

@numba.extending.register_model(FillableArrayType)
class FillableArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members= [("rawptr", numba.types.voidptr),
                  ("pyptr", numba.types.pyobject)]
        super(FillableArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(FillableArrayType)
def unbox_FillableArray(fillabletype, fillableobj, c):
    inner_obj = c.pyapi.object_getattr_string(fillableobj, "_fillablearray")
    rawptr_obj = c.pyapi.object_getattr_string(inner_obj, "_ptr")

    proxyout = c.context.make_helper(c.builder, fillabletype)
    proxyout.rawptr = c.pyapi.long_as_voidptr(rawptr_obj)
    proxyout.pyptr = inner_obj

    c.pyapi.decref(inner_obj)
    c.pyapi.decref(rawptr_obj)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(FillableArrayType)
def box_FillableArray(fillabletype, fillableval, c):
    import awkward1.highlevel
    FillableArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.highlevel.FillableArray))
    behavior_obj = c.pyapi.unserialize(c.pyapi.serialize_object(fillabletype.behavior))

    proxyin = c.context.make_helper(c.builder, fillabletype, fillableval)
    c.pyapi.incref(proxyin.pyptr)

    out = c.pyapi.call_method(FillableArray_obj, "_wrap", (proxyin.pyptr, behavior_obj))

    c.pyapi.decref(FillableArray_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(proxyin.pyptr)

    return out

def call(context, builder, fcn, args):
    fcntype = context.get_function_pointer_type(fcn.numbatype)
    fcnval = context.add_dynamic_addr(builder, fcn.numbatype.get_pointer(fcn), info=fcn.name)
    fcnptr = builder.bitcast(fcnval, fcntype)
    err = context.call_function_pointer(builder, fcnptr, args)
    with builder.if_then(builder.icmp_unsigned("!=", err, context.get_constant(numba.uint8, 0)), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, (fcn.name + " failed",))

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], FillableArrayType):
            return numba.intp(args[0])

@numba.extending.lower_builtin(len, FillableArrayType)
def lower_len(context, builder, sig, args):
    fillabletype, = sig.args
    fillableval, = args
    proxyin = context.make_helper(builder, fillabletype, fillableval)
    result = numba.cgutils.alloca_once(builder, context.get_value_type(numba.int64))
    call(context, builder, awkward1._numba.libawkward.FillableArray_length, (proxyin.rawptr, result))
    return awkward1._numba.castint(context, builder, numba.int64, numba.intp, builder.load(result))
