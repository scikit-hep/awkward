# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl
import numba.typing.ctypes_utils

import awkward1.layout
from .._numba import cpu, common

@numba.extending.typeof_impl.register(awkward1.layout.Identity)
def typeof(val, c):
    return IdentityType(numba.typeof(numpy.asarray(val)))

class IdentityType(common.ContentType):
    def __init__(self, arraytpe):
        super(IdentityType, self).__init__(name="IdentityType({0})".format(arraytpe.name))
        self.arraytpe = arraytpe

@numba.extending.register_model(IdentityType)
class IdentityModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("ref", common.RefType),
                   ("fieldloc", numba.types.List(numba.types.string)),
                   ("chunkdepth", common.IndexType),
                   ("indexdepth", common.IndexType),
                   ("array", common.IndexType[:, :])


("array", fe_type.arraytpe)]
        super(NumpyArrayModel, self).__init__(dmm, fe_type, members)
