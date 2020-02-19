# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

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

    def getitem_at(self):
        if self.arraytpe.ndim == 1:
            return self.arraytpe.dtype
        else:
            return numba.types.Array(self.arraytpe.dtype, self.arraytpe.ndim - 1, self.arraytpe.layout)

    def getitem_range(self):
        return self

    def getitem_field(self):
        raise TypeError("array has no fields")

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_at(self):
        return lower_getitem_at

    @property
    def lower_getitem_range(self):
        return lower_getitem_range

    @property
    def lower_getitem_field(self):
        raise AssertionError

@numba.extending.register_model(NumpyArrayType)
class NumpyArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe)]
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
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(array_obj)
    if tpe.identitiestpe != numba.none:
        raise NotImplementedError
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(NumpyArrayType)
def box(tpe, val, c):
    NumpyArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.NumpyArray))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    args = [array_obj]
    if tpe.identitiestpe != numba.none:
        raise NotImplementedError
    else:
        args.append(c.pyapi.make_none())
    args.append(awkward1._numba.util.items2dict_impl(c, tpe.parameters))
    for x in args:
        c.pyapi.decref(x)
    c.pyapi.decref(NumpyArray_obj)
    return out

@numba.extending.lower_builtin(len, NumpyArrayType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.arraytpe), (proxyin.array,))
