# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import json

import numba

import awkward1.layout
from .._numba import util

class LiteralTypeType(numba.types.Type):
    def __init__(self, literal_type):
        super(LiteralTypeType, self).__init__(name="ak::LiteralTypeType({0})".format(repr(literal_type)))
        self.literal_type = literal_type

@numba.extending.register_model(LiteralTypeType)
class LiteralTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        super(LiteralTypeModel, self).__init__(dmm, fe_type, [])

@numba.extending.unbox(LiteralTypeType)
def unbox(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(LiteralTypeType)
def box(tpe, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(tpe.literal_type))

def typeof_literaltype(literal_type):
    if isinstance(literal_type, awkward1.layout.ArrayType):
        literal_type = literal_type.type
    return LiteralTypeType(literal_type)

@numba.extending.typeof_impl.register(awkward1.layout.ArrayType)
def typeof_ArrayType(val, c):
    return numba.typeof(val.type)

@numba.extending.typeof_impl.register(awkward1.layout.UnknownType)
def typeof_UnknownType(val, c):
    return UnknownTypeType(val.parameters)

@numba.extending.typeof_impl.register(awkward1.layout.PrimitiveType)
def typeof_PrimitiveType(val, c):
    return PrimitiveTypeType(val.dtype, val.parameters)

@numba.extending.typeof_impl.register(awkward1.layout.RegularType)
def typeof_RegularType(val, c):
    return RegularTypeType(numba.typeof(val.type), val.parameters)

@numba.extending.typeof_impl.register(awkward1.layout.ListType)
def typeof_ListType(val, c):
    return ListTypeType(numba.typeof(val.type), val.parameters)

@numba.extending.typeof_impl.register(awkward1.layout.OptionType)
def typeof_OptionType(val, c):
    return OptionTypeType(numba.typeof(val.type), val.parameters)

@numba.extending.typeof_impl.register(awkward1.layout.UnionType)
def typeof_UnionType(val, c):
    return UnionTypeType([numba.typeof(x) for x in val.types], val.parameters)

@numba.extending.typeof_impl.register(awkward1.layout.RecordType)
def typeof_RecordType(val, c):
    return RecordTypeType([numba.typeof(x) for x in val.types], val.lookup, val.reverselookup, val.parameters)

# @numba.extending.typeof_impl.register(awkward1.layout.DressedType)
# def typeof_DressedType(val, c):
#     return DressedTypeType(numba.typeof(val.type), val.dress, val.parameters)

class TypeType(numba.types.Type):
    pass

class UnknownTypeType(TypeType):
    def __init__(self, parameters):
        super(UnknownTypeType, self).__init__(name="ak::UnknownTypeType(parameters={0})".format(json.dumps(parameters)))
        self.parameters = parameters

class PrimitiveTypeType(TypeType):
    def __init__(self, dtype, parameters):
        super(PrimitiveTypeType, self).__init__(name="ak::PrimitiveTypeType({0}, parameters={1})".format(dtype, json.dumps(parameters)))
        self.dtype = dtype
        self.parameters = parameters

class RegularTypeType(TypeType):
    def __init__(self, typetpe, parameters):
        super(RegularTypeType, self).__init__(name="ak::RegularTypeType({0}, parameters={1})".format(typetpe.name, json.dumps(parameters)))
        self.typetpe = typetpe
        self.parameters = parameters

class ListTypeType(TypeType):
    def __init__(self, typetpe, parameters):
        super(ListTypeType, self).__init__(name="ak::ListTypeType({0}, parameters={1})".format(typetpe.name, json.dumps(parameters)))
        self.typetpe = typetpe
        self.parameters = parameters

class OptionTypeType(TypeType):
    def __init__(self, typetpe, parameters):
        super(OptionTypeType, self).__init__(name="ak::OptionTypeType({0}, parameters={1})".format(typetpe.name, json.dumps(parameters)))
        self.typetpe = typetpe
        self.parameters = parameters

class UnionTypeType(TypeType):
    def __init__(self, typetpes, parameters):
        super(UnionTypeType, self).__init__(name="ak::UnionTypeType([{0}], parameters={1})".format(", ".join(x.name for x in typetpes), json.dumps(parameters)))
        self.typetpes = typetpes
        self.parameters = parameters

class RecordTypeType(TypeType):
    def __init__(self, typetpes, lookup, reverselookup, parameters):
        super(RecordTypeType, self).__init__(name="ak::RecordTypeType([{0}], {1}, {2}, parameters={3})".format(", ".join(x.name for x in typetpes), repr(lookup), repr(reverselookup), json.dumps(parameters)))
        self.typetpes = typetpes
        self.lookup = lookup
        self.reverselookup = reverselookup
        self.parameters = parameters

# class DressedTypeType(TypeType):
#     def __init__(self, typetpe, dress, parameters):
#         super(DressedTypeType, self).__init__(name="ak::DressedTypeType({0}, {1}, {2}, parameters={3})".format(typetpe.name, repr(dress), repr(parameters), json.dumps(parameters)))
#         self.typetpe = typetpe
#         self.dress = dress
#         self.parameters = parameters

@numba.extending.register_model(UnknownTypeType)
class UnknownTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        super(UnknownTypeModel, self).__init__(dmm, fe_type, [])

@numba.extending.register_model(PrimitiveTypeType)
class PrimitiveTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        super(PrimitiveTypeModel, self).__init__(dmm, fe_type, [])

@numba.extending.register_model(RegularTypeType)
class RegularTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("type", fe_type.typetpe),
                   ("size", numba.int64)]
        super(RegularTypeModel, self).__init__(dmm, fe_type, members)

@numba.extending.register_model(ListTypeType)
class ListTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("type", fe_type.typetpe)]
        super(ListTypeModel, self).__init__(dmm, fe_type, members)

@numba.extending.register_model(OptionTypeType)
class OptionTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("type", fe_type.typetpe)]
        super(OptionTypeModel, self).__init__(dmm, fe_type, members)

def field(i):
    return "f" + str(i)

@numba.extending.register_model(UnionTypeType)
class UnionTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = []
        for i, tpe in enumerate(fe_type.typetpes):
            members.append((field(i), tpe))
        super(UnionTypeModel, self).__init__(dmm, fe_type, members)

@numba.extending.register_model(RecordTypeType)
class RecordTypeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = []
        for i, tpe in enumerate(fe_type.typetpes):
            members.append((field(i), tpe))
        super(RecordTypeModel, self).__init__(dmm, fe_type, members)

# @numba.extending.register_model(DressedTypeType)
# class DressedTypeModel(numba.datamodel.models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [("type", fe_type.typetpe)]
#         super(DressedTypeModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(UnknownTypeType)
def unbox_UnknownType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(PrimitiveTypeType)
def unbox_PrimitiveType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(RegularTypeType)
def unbox_RegularType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    type_obj = c.pyapi.object_getattr_string(obj, "type")
    size_obj = c.pyapi.object_getattr_string(obj, "size")
    proxyout.type = c.pyapi.to_native_value(tpe.typetpe, type_obj).value
    proxyout.size = c.pyapi.to_native_value(numba.int64, size_obj).value
    c.pyapi.decref(type_obj)
    c.pyapi.decref(size_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(ListTypeType)
def unbox_ListType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    type_obj = c.pyapi.object_getattr_string(obj, "type")
    proxyout.type = c.pyapi.to_native_value(tpe.typetpe, type_obj).value
    c.pyapi.decref(type_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(OptionTypeType)
def unbox_OptionType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    type_obj = c.pyapi.object_getattr_string(obj, "type")
    proxyout.type = c.pyapi.to_native_value(tpe.typetpe, type_obj).value
    c.pyapi.decref(type_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(UnionTypeType)
def unbox_UnionType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    type_obj = c.pyapi.object_getattr_string(obj, "type")
    for i, t in enumerate(tpe.typetpes):
        i_obj = c.pyapi.long_from_longlong(c.context.get_constant(numba.int64, i))
        t_obj = c.pyapi.call_function_objargs(type_obj, (i_obj,))
        setattr(proxyout, field(i), c.pyapi.to_native_value(t, t_obj).value)
        c.pyapi.decref(i_obj)
        c.pyapi.decref(t_obj)
    c.pyapi.decref(type_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(RecordTypeType)
def unbox_RecordType(tpe, obj, c):
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    field_obj = c.pyapi.object_getattr_string(obj, "field")
    for i, t in enumerate(tpe.typetpes):
        i_obj = c.pyapi.long_from_longlong(c.context.get_constant(numba.int64, i))
        t_obj = c.pyapi.call_function_objargs(field_obj, (i_obj,))
        setattr(proxyout, field(i), c.pyapi.to_native_value(t, t_obj).value)
        c.pyapi.decref(i_obj)
        c.pyapi.decref(t_obj)
    c.pyapi.decref(field_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

# @numba.extending.unbox(DressedTypeType)
# def unbox_DressedType(tpe, obj, c):
#     proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
#     type_obj = c.pyapi.object_getattr_string(obj, "type")
#
#     proxyout.type = c.pyapi.to_native_value(tpe.typetpe, type_obj).value
#     c.pyapi.decref(type_obj)
#     is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#     return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(UnknownTypeType)
def box_UnknownType(tpe, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.UnknownType(tpe.parameters)))

@numba.extending.box(PrimitiveTypeType)
def box_PrimitiveType(tpe, val, c):
    return c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.PrimitiveType(tpe.dtype, tpe.parameters)))

def box_parameters(parameters, c):
    jsonloads_obj = c.pyapi.unserialize(c.pyapi.serialize_object(json.loads))
    paramstr_obj = c.pyapi.unserialize(c.pyapi.serialize_object(json.dumps(parameters)))
    param_obj = c.pyapi.call_function_objargs(jsonloads_obj, (paramstr_obj,))
    c.pyapi.decref(jsonloads_obj)
    c.pyapi.decref(paramstr_obj)
    return param_obj

@numba.extending.box(RegularTypeType)
def box_RegularType(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.RegularType))
    type_obj = c.pyapi.from_native_value(tpe.typetpe, proxyin.type, c.env_manager)
    size_obj = c.pyapi.long_from_longlong(proxyin.size)
    parameters_obj = box_parameters(tpe.parameters, c)
    out = c.pyapi.call_function_objargs(class_obj, (type_obj, size_obj, parameters_obj))
    c.pyapi.decref(class_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(size_obj)
    c.pyapi.decref(parameters_obj)
    return out

@numba.extending.box(ListTypeType)
def box_ListType(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.ListType))
    type_obj = c.pyapi.from_native_value(tpe.typetpe, proxyin.type, c.env_manager)
    parameters_obj = box_parameters(tpe.parameters, c)
    out = c.pyapi.call_function_objargs(class_obj, (type_obj, parameters_obj))
    c.pyapi.decref(class_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(parameters_obj)
    return out

@numba.extending.box(OptionTypeType)
def box_OptionType(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.OptionType))
    type_obj = c.pyapi.from_native_value(tpe.typetpe, proxyin.type, c.env_manager)
    parameters_obj = box_parameters(tpe.parameters, c)
    out = c.pyapi.call_function_objargs(class_obj, (type_obj, parameters_obj))
    c.pyapi.decref(class_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(parameters_obj)
    return out

@numba.extending.box(UnionTypeType)
def box_UnionType(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.UnionType))
    types_obj = c.pyapi.tuple_new(len(tpe.typetpes))
    for i, t in enumerate(tpe.typetpes):
        x_obj = c.pyapi.from_native_value(t, getattr(proxyin, field(i)), c.env_manager)
        c.pyapi.tuple_setitem(types_obj, i, x_obj)
    parameters_obj = box_parameters(tpe.parameters, c)
    out = c.pyapi.call_function_objargs(class_obj, (types_obj, parameters_obj))
    c.pyapi.decref(class_obj)
    c.pyapi.decref(types_obj)
    c.pyapi.decref(parameters_obj)
    return out

@numba.extending.box(RecordTypeType)
def box_RecordType(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.RecordType))
    types_obj = c.pyapi.tuple_new(len(tpe.typetpes))
    for i, t in enumerate(tpe.typetpes):
        x_obj = c.pyapi.from_native_value(t, getattr(proxyin, field(i)), c.env_manager)
        c.pyapi.tuple_setitem(types_obj, i, x_obj)
    parameters_obj = box_parameters(tpe.parameters, c)

    if tpe.lookup is None:
        out = c.pyapi.call_function_objargs(class_obj, (types_obj, parameters_obj))

    else:
        from_lookup_obj = c.pyapi.object_getattr_string(class_obj, "from_lookup")
        if tpe.lookup is None:
            lookup_obj = c.pyapi.make_none()
        else:
            lookup_obj = c.pyapi.dict_new(len(tpe.lookup))
            for key, fieldindex in tpe.lookup.items():
                key_obj = c.pyapi.unserialize(c.pyapi.serialize_object(key))
                fieldindex_obj = c.pyapi.unserialize(c.pyapi.serialize_object(fieldindex))
                c.pyapi.dict_setitem(lookup_obj, key_obj, fieldindex_obj)
                c.pyapi.decref(key_obj)
                c.pyapi.decref(fieldindex_obj)
        if tpe.reverselookup is None:
            reverselookup_obj = c.pyapi.make_none()
        else:
            reverselookup_obj = c.pyapi.list_new(c.context.get_constant(numba.intp, 0))
            for key in tpe.reverselookup:
                key_obj = c.pyapi.unserialize(c.pyapi.serialize_object(key))
                c.pyapi.list_append(reverselookup_obj, key_obj)
                c.pyapi.decref(key_obj)
        out = c.pyapi.call_function_objargs(from_lookup_obj, (types_obj, lookup_obj, reverselookup_obj, parameters_obj))
        c.pyapi.decref(from_lookup_obj)
        c.pyapi.decref(lookup_obj)
        c.pyapi.decref(reverselookup_obj)

    c.pyapi.decref(class_obj)
    c.pyapi.decref(types_obj)
    c.pyapi.decref(parameters_obj)
    return out

# @numba.extending.box(DressedTypeType)
# def box_DressedType(tpe, val, c):
#     proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
#     class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.DressedType))
#     type_obj = c.pyapi.from_native_value(tpe.typetpe, proxyin.type, c.env_manager)
#     dress_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tpe.dress))
#     args = c.pyapi.tuple_new(2)
#     c.pyapi.tuple_setitem(args, 0, type_obj)
#     c.pyapi.tuple_setitem(args, 1, dress_obj)
#     kwargs = c.pyapi.dict_new(len(tpe.parameters))
#     for n, x in tpe.parameters.items():
#         n_obj = c.pyapi.unserialize(c.pyapi.serialize_object(n))
#         x_obj = c.pyapi.unserialize(c.pyapi.serialize_object(x))
#         c.pyapi.dict_setitem(kwargs, n_obj, x_obj)
#         c.pyapi.decref(n_obj)
#         c.pyapi.decref(x_obj)
#     out = c.pyapi.call(class_obj, args, kwargs)
#     c.pyapi.decref(class_obj)
#     c.pyapi.decref(args)
#     c.pyapi.decref(kwargs)
#     return out

numba.extending.make_attribute_wrapper(RegularTypeType, "type", "type")
numba.extending.make_attribute_wrapper(ListTypeType, "type", "type")
numba.extending.make_attribute_wrapper(OptionTypeType, "type", "type")
# numba.extending.make_attribute_wrapper(DressedTypeType, "type", "type")
