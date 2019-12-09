# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numba

import awkward1.layout
from .._numba import util

# class AwkwardType(numba.types.Type):
#     def __init__(self, awkwardtype):
#         super(AwkwardType, self).__init__(name="ak::Type({})".format(repr(str(awkwardtype))))
#         self.awkwardtype = awkwardtype

# @numba.extending.typeof_impl.register(awkward1.layout.Type)
# def typeof(val, c):
#     return AwkwardType(val)

# @numba.extending.register_model(AwkwardType)
# class AwkwardTypeModel(numba.datamodel.models.StructModel):
#     members = []
#     super(AwkwardTypeModel, self).__init__(dmm, fe_type, members)




# class TypeType(numba.types.Type):
#     pass

# class ArrayType(TypeType):
#     def __init__(self, tpe):
#         super(ArrayType, self).__init__(name="ak::ArrayType({})".format(tpe.name))
#         self.tpe = tpe

# class ListType(TypeType):
#     def __init__(self, tpe):
#         super(ListType, self).__init__(name="ak::ListType({})".format(tpe.name))
#         self.tpe = tpe

# class RegularType(TypeType):
#     def __init__(self, tpe, size):
#         super(RegularType, self).__init__(name="ak::RegularType({}, {})".format(tpe.name, size))
#         self.tpe = tpe
#         self.size = size

# class OptionType(TypeType):
#     def __init__(self, tpe):
#         super(OptionType, self).__init__(name="ak::OptionType({})".format(tpe.name))
#         self.tpe = tpe

# class UnionType(TypeType):
#     def __init__(self, tpes):
#         super(UnionType, self).__init__(name="ak::UnionType({})".format(", ".join(x.name for x in tpes)))
#         self.tpes = tpes

# class RecordType(TypeType):
#     def __init__(self, tpes, lookup, reverselookup):
#         super(RecordType, self).__init__(name="ak::RecordType({}, {}, {})".format(", ".join(x.name for x in tpes), lookup, reverselookup))
#         self.tpes = tpes
#         self.lookup = lookup
#         self.reverselookup = reverselookup

# class DressedType(TypeType):
#     def __init__(self, tpe, dress, parameters):
#         super(RegularType, self).__init__(name="ak::DressedType({}, {}, {})".format(tpe.name, dress, parameters))
#         self.tpe = tpe
#         self.dress = dress
#         self.parameters = parameters

# class PrimitiveType(TypeType):
#     def __init__(self, dtype):
#         super(PrimitiveType, self).__init__(name="ak::PrimitiveType({})".format(dtype))
#         self.dtype = dtype

# class UnknownType(TypeType):
#     def __init__(self):
#         super(UnknownType, self).__init__(name="ak::UnknownType()")

# @numba.extending.typeof_impl.register(awkward1.layout.ArrayType)
# def typeof(val, c):
#     return ArrayType(val.type)

# @numba.extending.typeof_impl.register(awkward1.layout.ListType)
# def typeof(val, c):
#     return ListType(val.type)

# @numba.extending.typeof_impl.register(awkward1.layout.RegularType)
# def typeof(val, c):
#     return RegularType(val.type, val.size)

# @numba.extending.typeof_impl.register(awkward1.layout.OptionType)
# def typeof(val, c):
#     return OptionType(val.type)

# @numba.extending.typeof_impl.register(awkward1.layout.UnionType)
# def typeof(val, c):
#     return UnionType([numba.typeof(x) for x in val.types])

# @numba.extending.typeof_impl.register(awkward1.layout.RecordType)
# def typeof(val, c):
#     return RecordType([numba.typeof(x) for x in val.types], val.lookup, val.reverselookup)
