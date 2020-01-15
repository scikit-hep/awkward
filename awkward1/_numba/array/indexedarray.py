# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from ..._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.IndexedArray32)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedArrayU32)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedArray64)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedOptionArray32)
@numba.extending.typeof_impl.register(awkward1.layout.IndexedOptionArray64)
def typeof(val, c):
    import awkward1._numba.types
    return IndexedArrayType(numba.typeof(numpy.asarray(val.index)), numba.typeof(val.content), val.isoption, numba.typeof(val.identities), util.dict2parameters(val.parameters))

class IndexedArrayType(content.ContentType):
    def __init__(self, indextpe, contenttpe, isoption, identitiestpe, parameters):
        assert isinstance(parameters, tuple)
        super(IndexedArrayType, self).__init__(name="ak::Indexed{0}Array{1}{2}Type({3}, identities={4}, parameters={5})".format("Option" if isoption else "", "" if indextpe.dtype.signed else "U", indextpe.dtype.bitwidth, contenttpe.name, identitiestpe.name, util.parameters2str(parameters)))
        self.indextpe = indextpe
        self.contenttpe = contenttpe
        self.isoption = isoption
        self.identitiestpe = identitiestpe
        self.parameters = parameters

    @property
    def bitwidth(self):
        return self.indextpe.dtype.bitwidth

    @property
    def indexname(self):
        return ("" if self.indextpe.dtype.signed else "U") + str(self.indextpe.dtype.bitwidth)

    @property
    def ndim(self):
        return self.contenttpe.ndim

    def getitem_int(self):
        if self.isoption:
            return numba.types.optional(self.contenttpe.getitem_int())
        else:
            return self.contenttpe.getitem_int()

    def getitem_range(self):
        return self

    def getitem_str(self, key):
        return IndexedArrayType(self.indextpe, self.contenttpe.getitem_str(key), self.isoption, self.identitiestpe, ())

    def getitem_tuple(self, wheretpe):
        import awkward1._numba.array.listarray
        nexttpe = awkward1._numba.array.listarray.ListArrayType(util.index64tpe, util.index64tpe, self, numba.none, ())
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        import awkward1._numba.array.regulararray
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])

        if isinstance(headtpe, (numba.types.Integer, numba.types.SliceType, numba.types.Array)):
            if self.isoption:
                contenttpe = self.contenttpe.carry().getitem_next(wheretpe, isadvanced)
                return IndexedArrayType(self.indextpe, contenttpe, self.isoption, self.identitiestpe, self.parameters)
            else:
                return self.contenttpe.carry().getitem_next(wheretpe, isadvanced)

        elif isinstance(headtpe, numba.types.StringLiteral):
            return self.getitem_str(headtpe.literal_value).getitem_next(tailtpe, isadvanced)

        elif isinstance(headtpe, numba.types.EllipsisType):
            raise NotImplementedError("ellipsis")

        elif isinstance(headtpe, type(numba.typeof(numpy.newaxis))):
            raise NotImplementedError("newaxis")

        else:
            raise AssertionError(headtpe)

    def carry(self):
        return self

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

@numba.extending.register_model(IndexedArrayType)
class IndexedArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("index", fe_type.indextpe),
                   ("content", fe_type.contenttpe)]
        if fe_type.identitiestpe != numba.none:
            members.append(("identities", fe_type.identitiestpe))
        super(IndexedArrayModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(IndexedArrayType)
def unbox(tpe, obj, c):
    asarray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numpy.asarray))
    index_obj = c.pyapi.object_getattr_string(obj, "index")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    indexarray_obj = c.pyapi.call_function_objargs(asarray_obj, (index_obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.index = c.pyapi.to_native_value(tpe.indextpe, indexarray_obj).value
    proxyout.content = c.pyapi.to_native_value(tpe.contenttpe, content_obj).value
    c.pyapi.decref(asarray_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(indexarray_obj)
    if tpe.identitiestpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "identities")
        proxyout.identities = c.pyapi.to_native_value(tpe.identitiestpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(IndexedArrayType)
def box(tpe, val, c):
    if tpe.indexname == "64":
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index64))
        if tpe.isoption:
            IndexedArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexedOptionArray64))
        else:
            IndexedArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexedArray64))
    elif tpe.indexname == "32":
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Index32))
        if tpe.isoption:
            IndexedArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexedOptionArray32))
        else:
            IndexedArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexedArray32))
    elif tpe.indexname == "U32":
        Index_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexU32))
        if tpe.isoption:
            IndexedArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexedOptionArrayU32))
        else:
            IndexedArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.IndexedArrayU32))
    else:
        raise AssertionError("unrecognized index type: {0}".format(tpe.indexname))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    indexarray_obj = c.pyapi.from_native_value(tpe.indextpe, proxyin.index, c.env_manager)
    content_obj = c.pyapi.from_native_value(tpe.contenttpe, proxyin.content, c.env_manager)
    index_obj = c.pyapi.call_function_objargs(Index_obj, (indexarray_obj,))
    c.pyapi.decref(Index_obj)
    c.pyapi.decref(indexarray_obj)
    args = [index_obj, content_obj]
    if tpe.identitiestpe != numba.none:
        args.append(c.pyapi.from_native_value(tpe.identitiestpe, proxyin.identities, c.env_manager))
    else:
        args.append(c.pyapi.make_none())
    args.append(util.parameters2dict_impl(c, tpe.parameters))
    out = c.pyapi.call_function_objargs(IndexedArray_obj, args)
    for x in args:
        c.pyapi.decref(x)
    c.pyapi.decref(IndexedArray_obj)
    return out

@numba.extending.lower_builtin(len, IndexedArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    indexlen = numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe.indextpe), (proxyin.index,))
    return indexlen

@numba.extending.lower_builtin(operator.getitem, IndexedArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    indexval = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, tpe.indextpe.dtype(tpe.indextpe, wheretpe), (proxyin.index, whereval))

    outval = tpe.contenttpe.lower_getitem_int(context, builder, tpe.contenttpe.getitem_int()(tpe.contenttpe, tpe.indextpe.dtype), (proxyin.content, indexval))

    if tpe.isoption:
        output = context.make_helper(builder, tpe.getitem_int())
        with builder.if_else(builder.icmp_signed("<", indexval, context.get_constant(tpe.indextpe.dtype, 0))) as (isnone, isvalid):
            with isnone:
                output.valid = numba.cgutils.false_bit
                output.data = numba.cgutils.get_null_value(output.data.type)
            with isvalid:
                output.valid = numba.cgutils.true_bit
                output.data = tpe.contenttpe.lower_getitem_int(context, builder, tpe.contenttpe.getitem_int()(tpe.contenttpe, tpe.indextpe.dtype), (proxyin.content, indexval))
        return output._getvalue()

    else:
        return tpe.contenttpe.lower_getitem_int(context, builder, tpe.contenttpe.getitem_int()(tpe.contenttpe, tpe.indextpe.dtype), (proxyin.content, indexval))
