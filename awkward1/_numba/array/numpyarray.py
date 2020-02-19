# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import operator

import numpy
import numba

import awkward1.layout

import awkward1._numba.content
import awkward1._numba.util

class NumpyArrayType(awkward1._numba.content.ContentType):
    def __init__(self, arraytpe, identitiestpe, parameters):
        super(NumpyArrayType, self).__init__(name="awkward1.NumpyArrayType({0}, {1}, {2})".format(arraytpe.name, identitiestpe.name, awkward1._numba.util.items2str(parameters)))
        self.arraytpe = arraytpe
        self.identitiestpe = identitiestpe
        self.parameters = parameters

    def typeof_getitem_at(self):
        if self.arraytpe.ndim == 1:
            return self.arraytpe.dtype
        else:
            return NumpyArrayType(numba.types.Array(self.arraytpe.dtype, self.arraytpe.ndim - 1, self.arraytpe.layout), self.identitiestpe, self.parameters)

    def typeof_getitem_field(self):
        raise TypeError("array has no fields")

    @staticmethod
    def lower_getitem_at_nowrap(context, builder, sig, args):
        rettpe, (tpe, wheretpe) = sig.return_type, sig.args
        val, whereval = args
        # proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
        proxyin = context.make_helper(builder, tpe, val)

        if isinstance(rettpe, NumpyArrayType):
            proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
            proxyout.array = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, rettpe.arraytpe(tpe.arraytpe, wheretpe), (proxyin.array, whereval))
            proxyout.length = numba.targets.arrayobj.array_len(context, builder, numba.intp(rettpe.arraytpe), (proxyout.array,))
            if tpe.identitiestpe != numba.none:
                raise NotImplementedError
            return proxyout._getvalue()

        else:
            return numba.targets.arrayobj.getitem_arraynd_intp(context, builder, rettpe(tpe.arraytpe, wheretpe), (proxyin.array, whereval))

    @staticmethod
    def lower_getitem_range_nowrap(context, builder, tpe, val, whereval, length):
        # proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
        proxyin = context.make_helper(builder, tpe, val)
        proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
        proxyout.array = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.arraytpe(tpe.arraytpe, numba.types.slice2_type), (proxyin.array, whereval))
        proxyout.length = length
        return proxyout._getvalue()

@numba.extending.register_model(NumpyArrayType)
class NumpyArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe),
                   ("length", numba.intp)]
        if fe_type.identitiestpe != numba.none:
            raise NotImplementedError
        super(NumpyArrayModel, self).__init__(dmm, fe_type, members)

numba.extending.make_attribute_wrapper(NumpyArrayType, "array", "array")

@numba.extending.unbox(NumpyArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    array_obj = c.pyapi.call_function_objargs(asarray_obj, (obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    proxyout.length = numba.targets.arrayobj.array_len(c.context, c.builder, numba.intp(tpe.arraytpe), (proxyout.array,))
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(array_obj)
    if tpe.identitiestpe != numba.none:
        raise NotImplementedError
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(NumpyArrayType)
def box(tpe, val, c):
    NumpyArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.NumpyArray))
    # proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    proxyin = c.context.make_helper(c.builder, tpe, val)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    args = [array_obj]
    if tpe.identitiestpe != numba.none:
        raise NotImplementedError
    else:
        args.append(c.pyapi.make_none())
    args.append(awkward1._numba.util.items2dict_impl(c, tpe.parameters))
    out = c.pyapi.call_function_objargs(NumpyArray_obj, args)
    for x in args:
        c.pyapi.decref(x)
    c.pyapi.decref(NumpyArray_obj)
    return out
