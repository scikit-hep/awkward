# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numba

import awkward1.layout
from .._numba import cpu, util, content

class IteratorType(numba.types.common.SimpleIteratorType):
    def __init__(self, arraytpe):
        self.arraytpe = arraytpe
        super(IteratorType, self).__init__("ak::Iterator({0})".format(self.arraytpe.name), self.arraytpe.getitem_int())

@numba.typing.templates.infer
class ContentType_type_getiter(numba.typing.templates.AbstractTemplate):
    key = "getiter"

    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytpe, = args
            if isinstance(arraytpe, content.ContentType):
                return IteratorType(arraytpe)(arraytpe)

@numba.datamodel.registry.register_default(IteratorType)
class IteratorModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe),
                   ("length", numba.int64),
                   ("at", numba.types.EphemeralPointer(numba.int64))]
        super(IteratorModel, self).__init__(dmm, fe_type, members)

@numba.extending.lower_builtin("getiter", content.ContentType)
def lower_getiter(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyout = context.make_helper(builder, rettpe)
    proxyout.array = val
    proxyout.length = util.cast(context, builder, numba.intp, numba.int64, tpe.lower_len(context, builder, numba.intp(tpe), (val,)))
    proxyout.at = numba.cgutils.alloca_once_value(builder, context.get_constant(numba.int64, 0))
    if context.enable_nrt:
        context.nrt.incref(builder, tpe, val)
    return numba.targets.imputils.impl_ret_new_ref(context, builder, rettpe, proxyout._getvalue())

@numba.extending.lower_builtin("iternext", IteratorType)
@numba.targets.imputils.iternext_impl(numba.targets.imputils.RefType.BORROWED)
def lower_iternext(context, builder, sig, args, result):
    tpe, = sig.args
    val, = args

    proxyin = context.make_helper(builder, tpe, value=val)
    at = builder.load(proxyin.at)

    is_valid = builder.icmp_signed("<", at, proxyin.length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        result.yield_(tpe.arraytpe.lower_getitem_int(context, builder, tpe.yield_type(tpe.arraytpe, numba.int64), (proxyin.array, at)))
        nextat = numba.cgutils.increment_index(builder, at)
        builder.store(nextat, proxyin.at)
