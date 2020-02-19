# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import numba

import awkward1.highlevel
import awkward1._numba.util

import numba

class ArrayType(numba.types.Type):
    def __init__(self, layouttpe, behavior):
        super(ArrayType, self).__init__(name="awkward1.ArrayType({0}, {1})".format(layouttpe.name, awkward1._numba.util.dict2items(behavior)))
        self.layouttpe = layouttpe
        self.behavior = behavior

@numba.extending.register_model(ArrayType)
class ArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("layout", fe_type.layouttpe)]
        super(ArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(ArrayType)
def unbox(tpe, obj, c):
    layout_obj = c.pyapi.object_getattr_string(obj, "layout")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.layout = c.pyapi.to_native_value(tpe.layouttpe, layout_obj).value
    c.pyapi.decref(layout_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(ArrayType)
def box(tpe, val, c):
    Array_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.highlevel.Array))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    layout_obj = c.pyapi.from_native_value(tpe.layouttpe, proxyin.array, c.env_manager)
    args = [layout_obj]
    args.append(awkward1._numba.util.items2dict_impl(c, tpe.behavior))
    for x in args:
        c.pyapi.decref(x)
    c.pyapi.decref(Array_obj)
    return out
