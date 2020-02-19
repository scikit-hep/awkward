# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import operator

import numpy
import numba

class ContentType(numba.types.Type):
    def typeof_getitem_range(self):
        return ContentRangeType(self)

    def typeof_getitem_field(self):
        return ContentFieldType(self)

@numba.typing.templates.infer_global(len)
class type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytpe, = args
            if isinstance(arraytpe, ContentType):
                return numba.typing.templates.signature(numba.types.intp, arraytpe)

@numba.extending.lower_builtin(len, ContentType)
def lower_len(context, builder, sig, args):
    tpe, = sig.args
    val, = args
    # proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyin = context.make_helper(builder, tpe, val)
    return proxyin.length

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytpe, wheretpe = args
            if isinstance(arraytpe, ContentType):
                if isinstance(wheretpe, numba.types.Integer):
                    return numba.typing.templates.signature(arraytpe.typeof_getitem_at(), arraytpe, wheretpe)
                elif isinstance(wheretpe, numba.types.SliceType) and not wheretpe.has_step:
                    return numba.typing.templates.signature(arraytpe.typeof_getitem_range(), arraytpe, wheretpe)
                elif isinstance(wheretpe, numba.types.StringLiteral):
                    return numba.typing.templates.signature(arraytpe.typeof_getitem_field(), arraytpe, wheretpe)
                else:
                    raise TypeError("Awkward-Numba only supports int, start:stop, and \"field\" slices")

@numba.extending.lower_builtin(operator.getitem, ContentType, numba.types.Integer)
def lower_getitem_at(context, builder, sig, args):
    return sig.args[0].lower_getitem_at_nowrap(context, builder, sig, args)

@numba.extending.lower_builtin(operator.getitem, ContentType, numba.types.SliceType)
def lower_getitem_range(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    # proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, ref=val)
    proxyin = context.make_helper(builder, tpe, val)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe, val)

    proxyslicein = numba.cgutils.create_struct_proxy(wheretpe)(context, builder, value=whereval)
    numba.targets.slicing.fix_slice(builder, proxyslicein, proxyin.length)

    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.base = val   # proxyin._getpointer()
    proxyout.start = proxyslicein.start
    proxyout.stop = proxyslicein.stop
    return proxyout._getvalue()

class ContentRangeType(numba.types.Type):
    def __init__(self, basetpe):
        super(ContentRangeType, self).__init__(name="awkward1.ContentRangeType({0})".format(basetpe.name))
        self.basetpe = basetpe

    def typeof_getitem_at(self):
        return self.basetpe.typeof_getitem_at()

    def typeof_getitem_range(self):
        return self

    def typeof_getitem_field(self):
        raise NotImplementedError

@numba.extending.register_model(ContentRangeType)
class ContentRangeModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("base", fe_type.basetpe),   # numba.types.CPointer(fe_type.basetpe)),
                   ("start", numba.intp),
                   ("stop", numba.intp)]
        super(ContentRangeModel, self).__init__(dmm, fe_type, members)

@numba.extending.box(ContentRangeType)
def box_contentrange(tpe, val, c):
    # proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    proxyin = c.context.make_helper(c.builder, tpe, val)

    proxyslice = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(c.context, c.builder)
    proxyslice.start = proxyin.start
    proxyslice.stop = proxyin.stop
    proxyslice.step = c.context.get_constant(numba.intp, 1)
    length = c.builder.sub(proxyin.stop, proxyin.start)

    trimmed = tpe.basetpe.lower_getitem_range_nowrap(c.context, c.builder, tpe.basetpe, proxyin.base, proxyslice._getvalue(), length)   # c.builder.load(proxyin.base)

    return c.pyapi.from_native_value(tpe.basetpe, trimmed, c.env_manager)
