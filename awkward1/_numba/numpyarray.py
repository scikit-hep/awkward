# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, identity, content

@numba.extending.typeof_impl.register(awkward1.layout.NumpyArray)
def typeof(val, c):
    return NumpyArrayType(numba.typeof(numpy.asarray(val)), numba.typeof(val.id))

class NumpyArrayType(content.ContentType):
    def __init__(self, arraytpe, idtpe):
        super(NumpyArrayType, self).__init__(name="NumpyArrayType({0}, id={1})".format(arraytpe.name, idtpe.name))
        self.arraytpe = arraytpe
        self.idtpe = idtpe

    @property
    def ndim(self):
        return self.arraytpe.ndim

    def getitem_int(self):
        return self.getitem_tuple(numba.types.Tuple((numba.int64,)))

    def getitem_range(self):
        return self.getitem_tuple(numba.types.Tuple((numba.types.slice2_type,)))

    def getitem_tuple(self, wheretpe):
        return self.getitem_next(wheretpe, False)

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) > self.arraytpe.ndim:
            raise IndexError("too many indices for array")
        numreduce = sum(1 if isinstance(x, numba.types.Integer) else 0 for x in wheretpe.types)
        if numreduce < self.arraytpe.ndim:
            return NumpyArrayType(numba.types.Array(self.arraytpe.dtype, self.arraytpe.ndim - numreduce, self.arraytpe.layout), self.idtpe)
        elif numreduce == self.arraytpe.ndim:
            return self.arraytpe.dtype
        else:
            assert False

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_int(self):
        return lower_getitem

@numba.extending.register_model(NumpyArrayType)
class NumpyArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe)]
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(NumpyArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(NumpyArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    array_obj = c.pyapi.call_function_objargs(asarray_obj, (obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(array_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(NumpyArrayType)
def box(tpe, val, c):
    NumpyArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.NumpyArray))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        out = c.pyapi.call_function_objargs(NumpyArray_obj, (array_obj, id_obj))
        c.pyapi.decref(id_obj)
    else:
        out = c.pyapi.call_function_objargs(NumpyArray_obj, (array_obj,))
    c.pyapi.decref(NumpyArray_obj)
    c.pyapi.decref(array_obj)
    return out

@numba.extending.lower_builtin(len, NumpyArrayType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.arraytpe), (proxyin.array,))

@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.Integer)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.SliceType)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, NumpyArrayType, numba.types.BaseTuple)
def lower_getitem(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    if isinstance(rettpe, NumpyArrayType):
        signature = rettpe.arraytpe(tpe.arraytpe, wheretpe)
    else:
        signature = rettpe(tpe.arraytpe, wheretpe)

    if isinstance(wheretpe, numba.types.BaseTuple):
        out = numba.targets.arrayobj.getitem_array_tuple(context, builder, signature, (proxyin.array, whereval))
    elif isinstance(wheretpe, numba.types.Array):
        out = numba.targets.arrayobj.fancy_getitem_array(context, builder, signature, (proxyin.array, whereval))
    else:
        out = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, signature, (proxyin.array, whereval))

    if isinstance(rettpe, NumpyArrayType):
        proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
        proxyout.array = out
        return proxyout._getvalue()
    else:
        return out

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = NumpyArrayType

    def generic_resolve(self, tpe, attr):
        if attr == "id":
            if tpe.idtpe == numba.none:
                return numba.optional(identity.IdentityType(numba.int32[:, :]))
            else:
                return tpe.idtpe

@numba.extending.lower_getattr(NumpyArrayType, "id")
def lower_id(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if tpe.idtpe == numba.none:
        return context.make_optional_none(builder, identity.IdentityType(numba.int32[:, :]))
    else:
        if context.enable_nrt:
            context.nrt.incref(builder, tpe.idtpe, proxyin.id)
        return proxyin.id

#     @numba.typing.templates.bound_function("dummy1")
#     def resolve_dummy1(self, selftpe, args, kwargs):
#         if selftpe.arraytpe.dtype == numba.int32:
#             return numba.int32()
#
# dummy1tpe = numba.typing.ctypes_utils.make_function_type(cpu.kernels.dummy1)
#
# @numba.extending.lower_builtin("dummy1", NumpyArrayType)
# def lower_dummy1(context, builder, sig, args):
#     tpe, = sig.args
#     val, = args
#     proxy = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
#     inval = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, numba.int32(tpe.arraytpe, numba.intp), (proxy.array, context.get_constant(numba.intp, 0)))
#
#     ptrtpe = context.get_function_pointer_type(dummy1tpe)
#     ptrval = context.add_dynamic_addr(builder, dummy1tpe.get_pointer(cpu.kernels.dummy1), info="dummy1")
#     funcptr = builder.bitcast(ptrval, ptrtpe)
#
#     return context.call_function_pointer(builder, funcptr, [inval])
