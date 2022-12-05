# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
from awkward_cpp import libawkward

import awkward as ak

numpy = ak._nplikes.Numpy.instance()


dynamic_addrs = {}


def globalstring(context, builder, pyvalue):
    import llvmlite.ir.types

    if pyvalue not in dynamic_addrs:
        buf = dynamic_addrs[pyvalue] = numpy.array(pyvalue.encode("utf-8") + b"\x00")
        context.add_dynamic_addr(builder, buf.ctypes.data, info=f"str({repr(pyvalue)})")
    ptr = context.get_constant(numba.types.uintp, dynamic_addrs[pyvalue].ctypes.data)
    return builder.inttoptr(ptr, llvmlite.ir.PointerType(llvmlite.ir.IntType(8)))


class ArrayBuilderType(numba.types.Type):
    def __init__(self, behavior):
        super().__init__(
            name="ak.ArrayBuilderType({})".format(
                ak._connect.numba.arrayview.repr_behavior(behavior)
            )
        )
        self.behavior = behavior


@numba.extending.register_model(ArrayBuilderType)
class ArrayBuilderModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("rawptr", numba.types.voidptr), ("pyptr", numba.types.pyobject)]
        super().__init__(dmm, fe_type, members)


@numba.core.imputils.lower_constant(ArrayBuilderType)
def lower_const_ArrayBuilder(context, builder, arraybuildertype, arraybuilder):
    layout = arraybuilder._layout
    rawptr = context.get_constant(numba.intp, arraybuilder._layout._ptr)
    proxyout = context.make_helper(builder, arraybuildertype)
    proxyout.rawptr = builder.inttoptr(
        rawptr, context.get_value_type(numba.types.voidptr)
    )
    proxyout.pyptr = context.add_dynamic_addr(
        builder, id(layout), info=str(type(layout))
    )
    return proxyout._getvalue()


@numba.extending.unbox(ArrayBuilderType)
def unbox_ArrayBuilder(arraybuildertype, arraybuilderobj, c):
    inner_obj = c.pyapi.object_getattr_string(arraybuilderobj, "_layout")
    rawptr_obj = c.pyapi.object_getattr_string(inner_obj, "_ptr")

    proxyout = c.context.make_helper(c.builder, arraybuildertype)
    proxyout.rawptr = c.pyapi.long_as_voidptr(rawptr_obj)
    proxyout.pyptr = inner_obj

    c.pyapi.decref(inner_obj)
    c.pyapi.decref(rawptr_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(ArrayBuilderType)
def box_ArrayBuilder(arraybuildertype, arraybuilderval, c):
    ArrayBuilder_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(ak.highlevel.ArrayBuilder)
    )
    behavior_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(arraybuildertype.behavior)
    )

    proxyin = c.context.make_helper(c.builder, arraybuildertype, arraybuilderval)
    c.pyapi.incref(proxyin.pyptr)

    out = c.pyapi.call_method(ArrayBuilder_obj, "_wrap", (proxyin.pyptr, behavior_obj))

    c.pyapi.decref(ArrayBuilder_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(proxyin.pyptr)

    return out


def call(context, builder, fcn, args):
    numbatype = numba.core.typing.ctypes_utils.make_function_type(fcn)
    fcntype = context.get_function_pointer_type(numbatype)
    fcnval = context.add_dynamic_addr(
        builder, numbatype.get_pointer(fcn), info=fcn.name
    )
    fcnptr = builder.bitcast(fcnval, fcntype)
    err = context.call_function_pointer(builder, fcnptr, args)
    with builder.if_then(
        builder.icmp_unsigned("!=", err, context.get_constant(numba.uint8, 0)),
        likely=False,
    ):
        context.call_conv.return_user_exc(builder, ValueError, (fcn.name + " failed",))


@numba.core.typing.templates.infer_global(len)
class type_len(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], ArrayBuilderType)
        ):
            return numba.intp(args[0])


@numba.extending.lower_builtin(len, ArrayBuilderType)
def lower_len(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    result = numba.core.cgutils.alloca_once(
        builder, context.get_value_type(numba.int64)
    )
    call(
        context,
        builder,
        libawkward.ArrayBuilder_length,
        (proxyin.rawptr, result),
    )
    return ak._connect.numba.layout.castint(
        context, builder, numba.int64, numba.intp, builder.load(result)
    )


@numba.core.typing.templates.infer_getattr
class type_methods(numba.core.typing.templates.AttributeTemplate):
    key = ArrayBuilderType

    @numba.core.typing.templates.bound_function("clear")
    def resolve_clear(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for ArrayBuilder.clear")

    @numba.core.typing.templates.bound_function("null")
    def resolve_null(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for ArrayBuilder.null")

    @numba.core.typing.templates.bound_function("boolean")
    def resolve_boolean(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.Boolean)
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.boolean"
            )

    @numba.core.typing.templates.bound_function("integer")
    def resolve_integer(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.Integer)
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.integer"
            )

    @numba.core.typing.templates.bound_function("real")
    def resolve_real(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], (numba.types.Integer, numba.types.Float))
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError("wrong number or types of arguments for ArrayBuilder.real")

    @numba.core.typing.templates.bound_function("complex")
    def resolve_complex(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(
                args[0], (numba.types.Integer, numba.types.Float, numba.types.Complex)
            )
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.complex"
            )

    @numba.core.typing.templates.bound_function("datetime")
    def resolve_datetime(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], (numba.types.NPDatetime, numba.types.UnicodeType))
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.datetime"
            )

    @numba.core.typing.templates.bound_function("timedelta")
    def resolve_timedelta(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], (numba.types.NPTimedelta, numba.types.UnicodeType))
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.timedelta"
            )

    @numba.core.typing.templates.bound_function("string")
    def resolve_string(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], (numba.types.UnicodeType))
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.string"
            )

    @numba.core.typing.templates.bound_function("begin_list")
    def resolve_begin_list(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for ArrayBuilder.begin_list")

    @numba.core.typing.templates.bound_function("end_list")
    def resolve_end_list(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for ArrayBuilder.end_list")

    @numba.core.typing.templates.bound_function("begin_tuple")
    def resolve_begin_tuple(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.Integer)
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.begin_tuple"
            )

    @numba.core.typing.templates.bound_function("index")
    def resolve_index(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.Integer)
        ):
            return arraybuildertype(args[0])
        else:
            raise TypeError("wrong number or types of arguments for ArrayBuilder.index")

    @numba.core.typing.templates.bound_function("end_tuple")
    def resolve_end_tuple(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for ArrayBuilder.end_tuple")

    @numba.core.typing.templates.bound_function("begin_record")
    def resolve_begin_record(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        elif (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.StringLiteral)
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.begin_record"
            )

    @numba.core.typing.templates.bound_function("field")
    def resolve_field(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.StringLiteral)
        ):
            return arraybuildertype(args[0])
        else:
            raise TypeError("wrong number or types of arguments for ArrayBuilder.field")

    @numba.core.typing.templates.bound_function("end_record")
    def resolve_end_record(self, arraybuildertype, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return numba.types.none()
        else:
            raise TypeError("wrong number of arguments for ArrayBuilder.end_record")

    @numba.core.typing.templates.bound_function("append")
    def resolve_append(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(
                args[0],
                (
                    ak._connect.numba.arrayview.ArrayViewType,
                    ak._connect.numba.arrayview.RecordViewType,
                    numba.types.Boolean,
                    numba.types.Integer,
                    numba.types.Float,
                    numba.types.Complex,
                    numba.types.NPDatetime,
                    numba.types.NPTimedelta,
                    numba.types.UnicodeType,
                ),
            )
        ):
            return numba.types.none(args[0])
        elif (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.Optional)
            and isinstance(
                args[0].type,
                (
                    numba.types.Boolean,
                    numba.types.Integer,
                    numba.types.Float,
                    numba.types.Complex,
                    numba.types.NPDatetime,
                    numba.types.NPTimedelta,
                    numba.types.UnicodeType,
                ),
            )
        ):
            return numba.types.none(args[0])
        elif (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], numba.types.NoneType)
        ):
            return numba.types.none(args[0])
        else:
            if len(args) == 1 and arraybuildertype.behavior is not None:
                for key, lower in arraybuildertype.behavior.items():
                    if (
                        isinstance(key, tuple)
                        and len(key) == 3
                        and key[0] == "__numba_lower__"
                        and key[1] == ak.highlevel.ArrayBuilder.append
                        and (
                            args[0] == key[2]
                            or (
                                isinstance(key[2], type) and isinstance(args[0], key[2])
                            )
                        )
                    ):
                        numba.extending.lower_builtin(
                            "append", ArrayBuilderType, args[0]
                        )(lower)
                        return numba.types.none(args[0])

            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.append"
            )

    @numba.core.typing.templates.bound_function("extend")
    def resolve_extend(self, arraybuildertype, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], ak._connect.numba.arrayview.ArrayViewType)
        ):
            return numba.types.none(args[0])
        else:
            raise TypeError(
                "wrong number or types of arguments for ArrayBuilder.extend"
            )


@numba.extending.lower_builtin("clear", ArrayBuilderType)
def lower_clear(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_clear, (proxyin.rawptr,))
    return context.get_dummy_value()


@numba.extending.lower_builtin("null", ArrayBuilderType)
def lower_null(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_null, (proxyin.rawptr,))
    return context.get_dummy_value()


@numba.extending.lower_builtin("boolean", ArrayBuilderType, numba.types.Boolean)
def lower_boolean(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    x = builder.zext(xval, context.get_value_type(numba.uint8))
    call(context, builder, libawkward.ArrayBuilder_boolean, (proxyin.rawptr, x))
    return context.get_dummy_value()


@numba.extending.lower_builtin("integer", ArrayBuilderType, numba.types.Integer)
def lower_integer(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    x = ak._connect.numba.layout.castint(context, builder, xtype, numba.int64, xval)
    call(context, builder, libawkward.ArrayBuilder_integer, (proxyin.rawptr, x))
    return context.get_dummy_value()


@numba.extending.lower_builtin("real", ArrayBuilderType, numba.types.Integer)
@numba.extending.lower_builtin("real", ArrayBuilderType, numba.types.Float)
def lower_real(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    if isinstance(xtype, numba.types.Integer) and xtype.signed:
        x = builder.sitofp(xval, context.get_value_type(numba.types.float64))
    elif isinstance(xtype, numba.types.Integer):
        x = builder.uitofp(xval, context.get_value_type(numba.types.float64))
    elif xtype.bitwidth < 64:
        x = builder.fpext(xval, context.get_value_type(numba.types.float64))
    elif xtype.bitwidth > 64:
        x = builder.fptrunc(xval, context.get_value_type(numba.types.float64))
    else:
        x = xval
    call(context, builder, libawkward.ArrayBuilder_real, (proxyin.rawptr, x))
    return context.get_dummy_value()


@numba.extending.lower_builtin("complex", ArrayBuilderType, numba.types.Integer)
@numba.extending.lower_builtin("complex", ArrayBuilderType, numba.types.Float)
def lower_complex_from_integer_or_float(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)

    if isinstance(xtype, numba.types.Integer) and xtype.signed:
        z_real = builder.sitofp(xval, context.get_value_type(numba.types.float64))
        z_imag = z_real.type(0)
    elif isinstance(xtype, numba.types.Integer):
        z_real = builder.uitofp(xval, context.get_value_type(numba.types.float64))
        z_imag = z_real.type(0)
    elif xtype.bitwidth < 64:
        z_real = builder.fpext(xval, context.get_value_type(numba.types.float64))
    elif xtype.bitwidth > 64:
        z_real = builder.fptrunc(xval, context.get_value_type(numba.types.float64))
    else:
        z_real = xval
    z_imag = z_real.type(0)

    call(
        context,
        builder,
        libawkward.ArrayBuilder_complex,
        (proxyin.rawptr, z_real, z_imag),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("complex", ArrayBuilderType, numba.types.Complex)
def lower_complex(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)

    z = context.make_complex(builder, xtype, xval)
    z_real, z_imag = z.real, z.imag

    call(
        context,
        builder,
        libawkward.ArrayBuilder_complex,
        (proxyin.rawptr, z_real, z_imag),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("datetime", ArrayBuilderType, numba.types.NPDatetime)
def lower_datetime(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    unit = globalstring(context, builder, f"datetime64[{xtype.unit}]")
    call(
        context,
        builder,
        libawkward.ArrayBuilder_datetime,
        (proxyin.rawptr, xval, unit),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("timedelta", ArrayBuilderType, numba.types.NPTimedelta)
def lower_timedelta(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    unit = globalstring(context, builder, f"timedelta64[{xtype.unit}]")
    call(
        context,
        builder,
        libawkward.ArrayBuilder_timedelta,
        (proxyin.rawptr, xval, unit),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("string", ArrayBuilderType, numba.types.UnicodeType)
def lower_string(context, builder, sig, args):
    arraybuildertype, xtype = sig.args
    arraybuilderval, xval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)

    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    is_ok, out, length = pyapi.string_as_string_and_size(xval.value)
    length = ak._connect.numba.layout.castint(
        context, builder, numba.ssize_t, numba.int64, length
    )
    call(
        context,
        builder,
        libawkward.ArrayBuilder_string_length,
        (proxyin.rawptr, out, length),
    )

    pyapi.gil_release(gil)

    return context.get_dummy_value()


@numba.extending.lower_builtin("begin_list", ArrayBuilderType)
def lower_beginlist(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_beginlist, (proxyin.rawptr,))
    return context.get_dummy_value()


@numba.extending.lower_builtin("end_list", ArrayBuilderType)
def lower_endlist(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_endlist, (proxyin.rawptr,))
    return context.get_dummy_value()


@numba.extending.lower_builtin("begin_tuple", ArrayBuilderType, numba.types.Integer)
def lower_begintuple(context, builder, sig, args):
    arraybuildertype, numfieldstype = sig.args
    arraybuilderval, numfieldsval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    numfields = ak._connect.numba.layout.castint(
        context, builder, numfieldstype, numba.int64, numfieldsval
    )
    call(
        context,
        builder,
        libawkward.ArrayBuilder_begintuple,
        (proxyin.rawptr, numfields),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("index", ArrayBuilderType, numba.types.Integer)
def lower_index(context, builder, sig, args):
    arraybuildertype, indextype = sig.args
    arraybuilderval, indexval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    index = ak._connect.numba.layout.castint(
        context, builder, indextype, numba.int64, indexval
    )
    call(
        context,
        builder,
        libawkward.ArrayBuilder_index,
        (proxyin.rawptr, index),
    )
    return arraybuilderval


@numba.extending.lower_builtin("end_tuple", ArrayBuilderType)
def lower_endtuple(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_endtuple, (proxyin.rawptr,))
    return context.get_dummy_value()


@numba.extending.lower_builtin("begin_record", ArrayBuilderType)
def lower_beginrecord(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(
        context,
        builder,
        libawkward.ArrayBuilder_beginrecord,
        (proxyin.rawptr,),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin(
    "begin_record", ArrayBuilderType, numba.types.StringLiteral
)
def lower_beginrecord_field(context, builder, sig, args):
    arraybuildertype, nametype = sig.args
    arraybuilderval, nameval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    name = globalstring(context, builder, nametype.literal_value)
    call(
        context,
        builder,
        libawkward.ArrayBuilder_beginrecord_fast,
        (proxyin.rawptr, name),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("field", ArrayBuilderType, numba.types.StringLiteral)
def lower_field(context, builder, sig, args):
    arraybuildertype, keytype = sig.args
    arraybuilderval, keyval = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    key = globalstring(context, builder, keytype.literal_value)
    call(
        context,
        builder,
        libawkward.ArrayBuilder_field_fast,
        (proxyin.rawptr, key),
    )
    return arraybuilderval


@numba.extending.lower_builtin("end_record", ArrayBuilderType)
def lower_endrecord(context, builder, sig, args):
    (arraybuildertype,) = sig.args
    (arraybuilderval,) = args
    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_endrecord, (proxyin.rawptr,))
    return context.get_dummy_value()


@numba.extending.lower_builtin(
    "append", ArrayBuilderType, ak._connect.numba.arrayview.ArrayViewType
)
def lower_append_array(context, builder, sig, args):
    arraybuildertype, viewtype = sig.args
    arraybuilderval, viewval = args

    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(context, builder, libawkward.ArrayBuilder_beginlist, (proxyin.rawptr,))

    lower_extend_array(context, builder, sig, args)

    call(context, builder, libawkward.ArrayBuilder_endlist, (proxyin.rawptr,))

    return context.get_dummy_value()


@numba.extending.lower_builtin(
    "append", ArrayBuilderType, ak._connect.numba.arrayview.RecordViewType
)
def lower_append_record(context, builder, sig, args):
    arraybuildertype, recordviewtype = sig.args
    arraybuilderval, recordviewval = args

    recordviewproxy = context.make_helper(builder, recordviewtype, recordviewval)

    arrayviewproxy = context.make_helper(
        builder, recordviewtype.arrayviewtype, recordviewproxy.arrayview
    )
    atval = ak._connect.numba.layout.castint(
        context, builder, numba.intp, numba.int64, recordviewproxy.at
    )

    sharedptr = ak._connect.numba.layout.getat(
        context, builder, arrayviewproxy.sharedptrs, arrayviewproxy.pos
    )

    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    call(
        context,
        builder,
        libawkward.ArrayBuilder_append_nowrap,
        (
            proxyin.rawptr,
            builder.inttoptr(sharedptr, context.get_value_type(numba.types.voidptr)),
            atval,
        ),
    )
    return context.get_dummy_value()


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.Boolean)
def lower_append_bool(context, builder, sig, args):
    return lower_boolean(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.Integer)
def lower_append_int(context, builder, sig, args):
    return lower_integer(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.Float)
def lower_append_float(context, builder, sig, args):
    return lower_real(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.Complex)
def lower_append_complex(context, builder, sig, args):
    return lower_complex(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.NPDatetime)
def lower_append_datetime(context, builder, sig, args):
    return lower_datetime(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.NPTimedelta)
def lower_append_timedelta(context, builder, sig, args):
    return lower_timedelta(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.UnicodeType)
def lower_append_string(context, builder, sig, args):
    return lower_string(context, builder, sig, args)


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.Optional)
def lower_append_optional(context, builder, sig, args):
    arraybuildertype, opttype = sig.args
    arraybuilderval, optval = args

    optproxy = context.make_helper(builder, opttype, optval)
    validbit = numba.core.cgutils.as_bool_bit(builder, optproxy.valid)

    with builder.if_else(validbit) as (is_valid, is_not_valid):
        with is_valid:
            if isinstance(opttype.type, numba.types.Boolean):
                lower_boolean(
                    context,
                    builder,
                    numba.types.none(arraybuildertype, opttype.type),
                    (arraybuilderval, optproxy.data),
                )
            elif isinstance(opttype.type, numba.types.Integer):
                lower_integer(
                    context,
                    builder,
                    numba.types.none(arraybuildertype, opttype.type),
                    (arraybuilderval, optproxy.data),
                )
            elif isinstance(opttype.type, numba.types.Float):
                lower_real(
                    context,
                    builder,
                    numba.types.none(arraybuildertype, opttype.type),
                    (arraybuilderval, optproxy.data),
                )
            elif isinstance(opttype.type, numba.types.Complex):
                lower_complex(
                    context,
                    builder,
                    numba.types.none(arraybuildertype, opttype.type),
                    (arraybuilderval, optproxy.data),
                )
            elif isinstance(opttype.type, numba.types.UnicodeType):
                lower_string(
                    context,
                    builder,
                    numba.types.none(arraybuildertype, opttype.type),
                    (arraybuilderval, optproxy.data),
                )
            else:
                raise AssertionError(repr(opttype.type))

        with is_not_valid:
            lower_null(
                context,
                builder,
                numba.types.none(
                    arraybuildertype,
                ),
                (arraybuilderval,),
            )

    return context.get_dummy_value()


@numba.extending.lower_builtin("append", ArrayBuilderType, numba.types.NoneType)
def lower_append_none(context, builder, sig, args):
    return lower_null(context, builder, sig.return_type(sig.args[0]), (args[0],))


@numba.extending.lower_builtin(
    "extend", ArrayBuilderType, ak._connect.numba.arrayview.ArrayViewType
)
def lower_extend_array(context, builder, sig, args):
    arraybuildertype, viewtype = sig.args
    arraybuilderval, viewval = args

    viewproxy = context.make_helper(builder, viewtype, viewval)

    sharedptr = ak._connect.numba.layout.getat(
        context, builder, viewproxy.sharedptrs, viewproxy.pos
    )

    proxyin = context.make_helper(builder, arraybuildertype, arraybuilderval)
    with numba.core.cgutils.for_range(builder, viewproxy.stop, viewproxy.start) as loop:
        atval = ak._connect.numba.layout.castint(
            context, builder, numba.intp, numba.int64, loop.index
        )
        call(
            context,
            builder,
            libawkward.ArrayBuilder_append_nowrap,
            (
                proxyin.rawptr,
                builder.inttoptr(
                    sharedptr, context.get_value_type(numba.types.voidptr)
                ),
                atval,
            ),
        )

    return context.get_dummy_value()
