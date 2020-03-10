# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel
import awkward1.layout
import numpy

def register():
    import awkward1._connect._numba.arrayview
    import awkward1._connect._numba.layout
    import awkward1._connect._numba.builder

    awkward1.numba.ArrayViewType = awkward1._connect._numba.arrayview.ArrayViewType
    awkward1.numba.ArrayViewModel = awkward1._connect._numba.arrayview.ArrayViewModel
    awkward1.numba.RecordViewType = awkward1._connect._numba.arrayview.RecordViewType
    awkward1.numba.RecordViewModel = awkward1._connect._numba.arrayview.RecordViewModel
    awkward1.numba.ContentType = awkward1._connect._numba.layout.ContentType
    awkward1.numba.NumpyArrayType = awkward1._connect._numba.layout.NumpyArrayType
    awkward1.numba.RegularArrayType = awkward1._connect._numba.layout.RegularArrayType
    awkward1.numba.ListArrayType = awkward1._connect._numba.layout.ListArrayType
    awkward1.numba.IndexedArrayType = awkward1._connect._numba.layout.IndexedArrayType
    awkward1.numba.IndexedOptionArrayType = awkward1._connect._numba.layout.IndexedOptionArrayType
    awkward1.numba.RecordArrayType = awkward1._connect._numba.layout.RecordArrayType
    awkward1.numba.UnionArrayType = awkward1._connect._numba.layout.UnionArrayType
    awkward1.numba.ArrayBuilderType = awkward1._connect._numba.builder.ArrayBuilderType
    awkward1.numba.ArrayBuilderModel = awkward1._connect._numba.builder.ArrayBuilderModel

try:
    import numba
except ImportError:
    pass
else:
    @numba.extending.typeof_impl.register(awkward1.highlevel.Array)
    def typeof_Array(obj, c):
        return obj.numbatype

    @numba.extending.typeof_impl.register(awkward1.highlevel.Record)
    def typeof_Record(obj, c):
        return obj.numbatype

    @numba.extending.typeof_impl.register(awkward1.highlevel.ArrayBuilder)
    def typeof_ArrayBuilder(obj, c):
        return obj.numbatype

def repr_behavior(behavior):
    return repr(behavior)

def castint(context, builder, fromtype, totype, val):
    import llvmlite.ir.types

    if isinstance(fromtype, llvmlite.ir.types.IntType):
        if fromtype.width == 8:
            fromtype = numba.int8
        elif fromtype.width == 16:
            fromtype = numba.int16
        elif fromtype.width == 32:
            fromtype = numba.int32
        elif fromtype.width == 64:
            fromtype = numba.int64
    if not isinstance(fromtype, numba.types.Integer):
        raise AssertionError("unrecognized integer type: {0}".format(repr(fromtype)))

    if fromtype.bitwidth < totype.bitwidth:
        if fromtype.signed:
            return builder.sext(val, context.get_value_type(totype))
        else:
            return builder.zext(val, context.get_value_type(totype))
    elif fromtype.bitwidth > totype.bitwidth:
        return builder.trunc(val, context.get_value_type(totype))
    else:
        return val
