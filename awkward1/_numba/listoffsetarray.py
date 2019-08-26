# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from .._numba import cpu, common

@numba.extending.typeof_impl.register(awkward1.layout.ListOffsetArray)
def typeof(val, c):
    return ListOffsetArrayType(numba.typeof(numpy.asarray(val.offsets)), numba.typeof(val.content))

class ListOffsetArrayType(common.ContentType):
    def __init__(self, offsetstpe, contenttpe):
        super(ListOffsetArrayType, self).__init__(name="ListOffsetArrayType({0}, {1})".format(offsetstpe.name, contenttpe.name))
        self.offsetstpe = offsetstpe
        self.contenttpe = contenttpe

    def getitem(self, wheretpe):
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])
        if isinstance(headtpe, numba.types.Integer):
            return self.contenttpe.getitem(tailtpe)
        else:
            return self.getitem(tailtpe)

@numba.extending.register_model(ListOffsetArrayType)
class ListOffsetArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("offsets", fe_type.offsetstpe),
                   ("content", fe_type.contenttpe)]
        super(ListOffsetArrayModel, self).__init__(dmm, fe_type, members)

# @numba.extending.unbox(ListOffsetArrayType)
# def unbox(tpe, obj, c):
#     asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
#     array_obj = c.pyapi.call_function_objargs(asarray_obj, (obj,))
