# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from ..._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.RecordArray)
def typeof(val, c):
    return RecordArrayType([numba.typeof(x) for x in val.values()], val.lookup, numba.typeof(val.id))

@numba.extending.typeof_impl.register(awkward1.layout.Record)
def typeof(val, c):
    return RecordType(RecordArrayType(val.recordarray))

class RecordArrayType(content.ContentType):
    def __init__(self, contenttpes, lookup, idtpe):
        super(RecordArrayType, self).__init__(name="RecordArrayType([{}], {}, id={})".format(", ".join(x.name for x in contenttpes), lookup, idtpe.name))
        self.contenttpes = contenttpes
        self.lookup = lookup
        self.idtpe = idtpe

    @property
    def ndim(self):
        return 1

    def getitem_int(self):
        return RecordType(self)

    def getitem_range(self):
        return self

    def getitem_tuple(self, wheretpe):
        nexttpe = RegularArrayType(self, numba.none)
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])

        raise NotImplementedError

    def carry(self):
        return RecordArrayType([x.carry() for x in self.contenttpes], self.lookup, self.idtpe)

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_nothing(self):
        return content.lower_getitem_nothing

    @property
    def lower_getitem_int(self):
        return lower_getitem_int

    @property
    def lower_getitem_range(self):
        return lower_getitem_range

    @property
    def lower_getitem_next(self):
        return lower_getitem_next

    @property
    def lower_carry(self):
        return lower_carry

class RecordType(numba.types.Type):
    def __init__(self, arraytpe):
        self.arraytpe = arraytpe
        super(RecordType, self).__init__("Record({})".format(self.arraytpe.name))

@numba.extending.register_model(RecordArrayType)
class RecordArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("length", numba.int64)]
        for i, x in enumerate(contenttpes):
            members.append((str(i), x))
        if fe_type.idtype != numba.none:
            members.append(("id", fe_type.idtpe))
        super(RecordArrayModel, self).__init__(dmm, fe_type, members)

@numba.datamodel.registry.register_default(RecordType)
class RecordModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe),
                   ("at", numba.int64)]
        super(RecordModel, self).__init__(dmm, fe_type, members)
