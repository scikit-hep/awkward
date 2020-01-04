# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from ..._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.RecordArray)
def typeof(val, c):
    return RecordArrayType([numba.typeof(x) for x in val.fields()], None if val.istuple else tuple(val.keys()), numba.typeof(val.identities), util.dict2parameters(val.parameters))

@numba.extending.typeof_impl.register(awkward1.layout.Record)
def typeof(val, c):
    return RecordType(numba.typeof(val.array))

class RecordArrayType(content.ContentType):
    def __init__(self, contenttpes, keys, identitiestpe, parameters):
        assert isinstance(parameters, tuple)
        super(RecordArrayType, self).__init__(name="ak::RecordArrayType([{0}], {1}, identities={2}, parameters={3})".format(", ".join(x.name for x in contenttpes), keys, identitiestpe.name, util.parameters2str(parameters)))
        self.contenttpes = contenttpes
        self.keys = keys
        self.identitiestpe = identitiestpe
        self.parameters = parameters

    @property
    def istuple(self):
        return self.keys is None

    @property
    def numfields(self):
        return len(self.contenttpes)

    @property
    def ndim(self):
        return 1

    def getitem_int(self):
        return RecordType(self)

    def getitem_range(self):
        return self

    def getitem_str(self, key):
        return self.contenttpes[awkward1._util.key2index(self.keys, key)]

    def getitem_tuple(self, wheretpe):
        import awkward1._numba.array.regulararray
        nexttpe = awkward1._numba.array.regulararray.RegularArrayType(self, numba.none, ())
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])

        if isinstance(headtpe, numba.types.StringLiteral):
            index = awkward1._util.key2index(self.keys, headtpe.literal_value)
            nexttpe = self.contenttpes[index]

        else:
            contenttpes = []
            for t in self.contenttpes:
                contenttpes.append(t.getitem_next(numba.types.Tuple((headtpe,)), isadvanced))
            nexttpe = RecordArrayType(contenttpes, self.keys, numba.none, ())

        return nexttpe.getitem_next(tailtpe, isadvanced)

    def carry(self):
        return RecordArrayType([x.carry() for x in self.contenttpes], self.keys, self.identitiestpe, self.parameters)

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
    def lower_getitem_str(self):
        return lower_getitem_str

    @property
    def lower_getitem_next(self):
        return lower_getitem_next

    @property
    def lower_carry(self):
        return lower_carry

class RecordType(numba.types.Type):
    def __init__(self, arraytpe):
        self.arraytpe = arraytpe
        super(RecordType, self).__init__("Record({0})".format(self.arraytpe.name))
        assert isinstance(arraytpe, RecordArrayType)

    @property
    def istuple(self):
        return self.arraytpe.istuple

    def getitem_str(self, key):
        outtpe = self.arraytpe.getitem_str(key)
        return outtpe.getitem_int()

    def getitem_tuple(self, wheretpe):
        nextwheretpe = numba.types.Tuple((numba.int64,) + wheretpe.types)
        return self.arraytpe.getitem_tuple(nextwheretpe)

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem_record(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            tpe, wheretpe = args

            if isinstance(tpe, RecordType):
                original_wheretpe = wheretpe
                if isinstance(wheretpe, numba.types.Integer):
                    raise TypeError("Record[int]")
                if isinstance(wheretpe, numba.types.SliceType):
                    raise TypeError("Record[slice]")
                if isinstance(wheretpe, numba.types.StringLiteral):
                    return numba.typing.templates.signature(tpe.getitem_str(wheretpe.literal_value), tpe, original_wheretpe)

                if not isinstance(wheretpe, numba.types.BaseTuple):
                    wheretpe = numba.types.Tuple((wheretpe,))

                wheretpe = util.typing_regularize_slice(wheretpe)
                content.type_getitem.check_slice_types(wheretpe)

                return numba.typing.templates.signature(tpe.getitem_tuple(wheretpe), tpe, original_wheretpe)

def field(i):
    return "f" + str(i)

@numba.extending.register_model(RecordArrayType)
class RecordArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("length", numba.int64)]
        for i, x in enumerate(fe_type.contenttpes):
            members.append((field(i), x))
        if fe_type.identitiestpe != numba.none:
            members.append(("identities", fe_type.identitiestpe))
        super(RecordArrayModel, self).__init__(dmm, fe_type, members)

@numba.datamodel.registry.register_default(RecordType)
class RecordModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe),
                   ("at", numba.int64)]
        super(RecordModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(RecordArrayType)
def unbox(tpe, obj, c):
    len_obj = c.pyapi.unserialize(c.pyapi.serialize_object(len))
    length_obj = c.pyapi.call_function_objargs(len_obj, (obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.length = c.pyapi.to_native_value(numba.int64, length_obj).value
    c.pyapi.decref(len_obj)
    c.pyapi.decref(length_obj)
    field_obj = c.pyapi.object_getattr_string(obj, "field")
    for i, t in enumerate(tpe.contenttpes):
        i_obj = c.pyapi.long_from_longlong(c.context.get_constant(numba.int64, i))
        x_obj = c.pyapi.call_function_objargs(field_obj, (i_obj,))
        setattr(proxyout, field(i), c.pyapi.to_native_value(t, x_obj).value)
        c.pyapi.decref(i_obj)
        c.pyapi.decref(x_obj)
    c.pyapi.decref(field_obj)
    if tpe.identitiestpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "identities")
        proxyout.identities = c.pyapi.to_native_value(tpe.identitiestpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(RecordType)
def unbox_record(tpe, obj, c):
    array_obj = c.pyapi.object_getattr_string(obj, "array")
    at_obj = c.pyapi.object_getattr_string(obj, "at")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    proxyout.at = c.pyapi.to_native_value(numba.int64, at_obj).value
    if c.context.enable_nrt:
        c.context.nrt.incref(c.builder, tpe.arraytpe, proxyout.array)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(at_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(RecordArrayType)
def box(tpe, val, c):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    args = []
    if tpe.identitiestpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.identitiestpe, proxyin.identities, c.env_manager)
        args.append(id_obj)
    else:
        args.append(c.pyapi.make_none())
    args.append(util.parameters2dict_impl(c, tpe.parameters))

    if len(tpe.contenttpes) == 0:
        RecordArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.RecordArray))
        length_obj = c.pyapi.long_from_longlong(proxyin.length)
        istuple_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tpe.istuple))
        out = c.pyapi.call_function_objargs(RecordArray_obj, [length_obj, istuple_obj] + args)
        c.pyapi.decref(RecordArray_obj)
        c.pyapi.decref(length_obj)
        c.pyapi.decref(istuple_obj)

    else:
        RecordArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.RecordArray))
        contents_obj = c.pyapi.list_new(c.context.get_constant(numba.intp, 0))
        for i, t in enumerate(tpe.contenttpes):
            x_obj = c.pyapi.from_native_value(t, getattr(proxyin, field(i)), c.env_manager)
            c.pyapi.list_append(contents_obj, x_obj)
            c.pyapi.decref(x_obj)
        keys_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tpe.keys))
        out = c.pyapi.call_function_objargs(RecordArray_obj, [contents_obj, keys_obj] + args)
        c.pyapi.decref(RecordArray_obj)
        c.pyapi.decref(contents_obj)
        c.pyapi.decref(keys_obj)

    for x in args:
        c.pyapi.decref(x)

    return out

@numba.extending.box(RecordType)
def box_record(tpe, val, c):
    Record_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Record))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    at_obj = c.pyapi.from_native_value(numba.int64, proxyin.at, c.env_manager)
    out = c.pyapi.call_function_objargs(Record_obj, (array_obj, at_obj))
    c.pyapi.decref(Record_obj)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(at_obj)
    return out

@numba.extending.lower_builtin(len, RecordArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return util.cast(context, builder, numba.int64, numba.intp, proxyin.length)

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.array = val
    proxyout.at = util.cast(context, builder, wheretpe, numba.int64, whereval)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe, val)
    return numba.targets.imputils.impl_ret_new_ref(context, builder, rettpe, proxyout._getvalue())

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    import awkward1._numba.identities

    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    proxyslicein = context.make_helper(builder, wheretpe, value=whereval)
    numba.targets.slicing.guard_invalid_slice(context, builder, wheretpe, proxyslicein)
    numba.targets.slicing.fix_slice(builder, proxyslicein, util.cast(context, builder, numba.int64, numba.intp, proxyin.length))
    proxysliceout = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxysliceout.start = proxyslicein.start
    proxysliceout.stop = proxyslicein.stop
    proxysliceout.step = proxyslicein.step
    sliceout = proxysliceout._getvalue()

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.length = util.cast(context, builder, numba.intp, numba.int64, builder.sub(proxyslicein.stop, proxyslicein.start))
    for i, t in enumerate(tpe.contenttpes):
        setattr(proxyout, field(i), t.lower_getitem_range(context, builder, t.getitem_range()(t, numba.types.slice2_type), (getattr(proxyin, field(i)), sliceout)))
    if tpe.identitiestpe != numba.none:
        proxyout.identities = awkward1._numba.identities.lower_getitem_any(context, builder, tpe.identitiestpe, wheretpe, proxyin.identities, whereval)

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.StringLiteral)
def lower_getitem_str(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    index = awkward1._util.key2index(tpe.keys, wheretpe.literal_value)

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    out = getattr(proxyin, field(index))
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, RecordType, numba.types.StringLiteral)
def lower_getitem_str_record(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    outtpe = tpe.arraytpe.getitem_str(wheretpe.literal_value)
    outval = lower_getitem_str(context, builder, outtpe(tpe.arraytpe, wheretpe), (proxyin.array, whereval))
    return outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, proxyin.at))

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    return content.lower_getitem_tuple(context, builder, sig, args)

@numba.extending.lower_builtin(operator.getitem, RecordType, numba.types.BaseTuple)
def lower_getitem_tuple_record(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    nextwheretpe = numba.types.Tuple((numba.int64,) + wheretpe.types)
    nextwhereval = context.make_tuple(builder, nextwheretpe, (proxyin.at,) + numba.cgutils.unpack_tuple(builder, whereval))

    return lower_getitem_tuple(context, builder, rettpe(tpe.arraytpe, nextwheretpe), (proxyin.array, nextwhereval))

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.List)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.ArrayCompatible)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.EllipsisType)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, type(numba.typeof(numpy.newaxis)))
def lower_getitem_other(context, builder, sig, args):
    return content.lower_getitem_other(context, builder, sig, args)

def lower_getitem_next(context, builder, arraytpe, wheretpe, arrayval, whereval, advanced):
    if len(wheretpe.types) == 0:
        return arrayval

    headtpe = wheretpe.types[0]
    tailtpe = numba.types.Tuple(wheretpe.types[1:])
    headval = numba.cgutils.unpack_tuple(builder, whereval)[0]
    tailval = context.make_tuple(builder, tailtpe, numba.cgutils.unpack_tuple(builder, whereval)[1:])

    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)

    if isinstance(headtpe, numba.types.StringLiteral):
        index = awkward1._util.key2index(arraytpe.keys, headtpe.literal_value)
        nexttpe = arraytpe.contenttpes[index]
        nextval = getattr(proxyin, field(index))

    else:
        nexttpe = RecordArrayType([t.getitem_next(numba.types.Tuple((headtpe,)), advanced is not None) for t in arraytpe.contenttpes], arraytpe.keys, numba.none, arraytpe.parameters if util.preserves_type(headtpe, advanced is not None) else ())
        proxyout = numba.cgutils.create_struct_proxy(nexttpe)(context, builder)
        proxyout.length = proxyin.length
        wrappedheadtpe = numba.types.Tuple((headtpe,))
        wrappedheadval = context.make_tuple(builder, wrappedheadtpe, (headval,))

        for i, t in enumerate(arraytpe.contenttpes):
            setattr(proxyout, field(i), t.lower_getitem_next(context, builder, t, wrappedheadtpe, getattr(proxyin, field(i)), wrappedheadval, advanced))
        nextval = proxyout._getvalue()

    rettpe = nexttpe.getitem_next(tailtpe, advanced is not None)
    return rettpe.lower_getitem_next(context, builder, nexttpe, tailtpe, nextval, tailval, advanced)

def lower_carry(context, builder, arraytpe, carrytpe, arrayval, carryval):
    import awkward1._numba.identities
    rettpe = arraytpe.carry()
    proxyin = numba.cgutils.create_struct_proxy(arraytpe)(context, builder, value=arrayval)
    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.length = util.arraylen(context, builder, carrytpe, carryval, totpe=numba.int64)
    for i, t in enumerate(arraytpe.contenttpes):
        setattr(proxyout, field(i), t.lower_carry(context, builder, t, carrytpe, getattr(proxyin, field(i)), carryval))
    if rettpe.identitiestpe != numba.none:
        proxyout.identities = awkward1._numba.identities.lower_getitem_any(context, builder, rettpe.identitiestpe, carrytpe, proxyin.identities, carryval)
    return proxyout._getvalue()

@numba.typing.templates.infer_getattr
class type_methods(numba.typing.templates.AttributeTemplate):
    key = RecordArrayType

    def generic_resolve(self, tpe, attr):
        if attr == "identities":
            if tpe.identitiestpe == numba.none:
                return numba.optional(identity.IdentitiesType(numba.int32[:, :]))
            else:
                return tpe.identitiestpe

@numba.extending.lower_getattr(RecordArrayType, "identities")
def lower_identities(context, builder, tpe, val):
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    if tpe.identitiestpe == numba.none:
        return context.make_optional_none(builder, identity.IdentitiesType(numba.int32[:, :]))
    else:
        if context.enable_nrt:
            context.nrt.incref(builder, tpe.identitiestpe, proxyin.identities)
        return proxyin.identities
