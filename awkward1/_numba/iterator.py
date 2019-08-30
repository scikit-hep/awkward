# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from .._numba import cpu, util, content

class IteratorType(numba.types.common.SimpleIteratorType):
    wheretpe = util.ChunkOffsetType

    def __init__(self, arraytpe):
        self.arraytpe = arraytpe
        super(IteratorType, self).__init__("iter({0})".format(self.arraytpe.name), self.arraytpe.getitem(numba.types.Tuple((self.wheretpe,))))

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
                   ("where", numba.types.EphemeralPointer(fe_type.wheretpe))]
        super(IteratorModel, self).__init__(dmm, fe_type, members)

@numba.extending.lower_builtin("getiter", content.ContentType)
def lower_getiter(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyout = context.make_helper(builder, rettpe)
    proxyout.array = val
    proxyout.where = numba.cgutils.alloca_once_value(builder, context.get_constant(rettpe.wheretpe, 0))
    if context.enable_nrt:
        context.nrt.incref(builder, tpe, val)
    return numba.targets.imputils.impl_ret_new_ref(context, builder, rettpe, proxyout._getvalue())

@numba.extending.lower_builtin("iternext", IteratorType)
@numba.targets.imputils.iternext_impl(numba.targets.imputils.RefType.BORROWED)
def lower_iternext(context, builder, sig, args, result):
    print("itenext BEGIN")
    tpe, = sig.args
    val, = args

    print("ONE")

    iterator_proxy = context.make_helper(builder, tpe, value=val)
    array_proxy = numba.cgutils.create_struct_proxy(tpe.arraytpe)(context, builder, value=iterator_proxy.array)

    print("TWO", iterator_proxy.where)

    where = builder.load(iterator_proxy.where)

    print("TWO.FIVE", tpe.arraytpe.lower_len)

    is_valid = builder.icmp_signed("<", where, tpe.arraytpe.__module__.lower_len(context, builder, numba.types.intp(tpe.arraytype), (iterator_proxy.array,)))

    print("TWO.SIX")

    result.set_valid(is_valid)

    print("THREE")

    with builder.if_then(is_valid, likely=True):
        result.yield_(tpe.arraytpe.__module__.lower_getitem_int(context, builder, tpe.yield_type(tpe.arraytpe, tpe.wheretpe), (proxyin.array, where)))
        nextwhere = numba.cgutils.increment_index(builder, where)
        builder.store(nextwhere, iterator_proxy.where)
    print("itenext END")
