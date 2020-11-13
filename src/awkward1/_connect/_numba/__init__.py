# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import distutils.version

import awkward1.highlevel
import awkward1.layout
import awkward1._util


checked_version = False


def register_and_check():
    global checked_version
    try:
        import numba
    except ImportError:
        raise ImportError(
            """install the 'numba' package with:

    pip install numba --upgrade

or

    conda install numba"""
        )
    else:
        if not checked_version and distutils.version.LooseVersion(
            numba.__version__
        ) < distutils.version.LooseVersion("0.50"):
            raise ImportError(
                "awkward1 can only work with numba 0.50 or later "
                "(you have version {0})".format(numba.__version__)
            )
        checked_version = True
        register()


def register():
    import numba
    import awkward1._connect._numba.arrayview
    import awkward1._connect._numba.layout
    import awkward1._connect._numba.builder

    if hasattr(awkward1.numba, "ArrayViewType"):
        return

    n = awkward1.numba
    n.ArrayViewType = awkward1._connect._numba.arrayview.ArrayViewType
    n.ArrayViewModel = awkward1._connect._numba.arrayview.ArrayViewModel
    n.RecordViewType = awkward1._connect._numba.arrayview.RecordViewType
    n.RecordViewModel = awkward1._connect._numba.arrayview.RecordViewModel
    n.ContentType = awkward1._connect._numba.layout.ContentType
    n.NumpyArrayType = awkward1._connect._numba.layout.NumpyArrayType
    n.RegularArrayType = awkward1._connect._numba.layout.RegularArrayType
    n.ListArrayType = awkward1._connect._numba.layout.ListArrayType
    n.IndexedArrayType = awkward1._connect._numba.layout.IndexedArrayType
    n.IndexedOptionArrayType = awkward1._connect._numba.layout.IndexedOptionArrayType
    n.ByteMaskedArrayType = awkward1._connect._numba.layout.ByteMaskedArrayType
    n.BitMaskedArrayType = awkward1._connect._numba.layout.BitMaskedArrayType
    n.UnmaskedArrayType = awkward1._connect._numba.layout.UnmaskedArrayType
    n.RecordArrayType = awkward1._connect._numba.layout.RecordArrayType
    n.UnionArrayType = awkward1._connect._numba.layout.UnionArrayType
    n.ArrayBuilderType = awkward1._connect._numba.builder.ArrayBuilderType
    n.ArrayBuilderModel = awkward1._connect._numba.builder.ArrayBuilderModel

    @numba.extending.typeof_impl.register(awkward1.highlevel.Array)
    def typeof_Array(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(awkward1.highlevel.Record)
    def typeof_Record(obj, c):
        return obj.numba_type

    @numba.extending.typeof_impl.register(awkward1.highlevel.ArrayBuilder)
    def typeof_ArrayBuilder(obj, c):
        return obj.numba_type


def repr_behavior(behavior):
    return repr(behavior)


def castint(context, builder, fromtype, totype, val):
    import numba
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
        raise AssertionError(
            "unrecognized integer type: {0}".format(repr(fromtype))
            + awkward1._util.exception_suffix(__file__)
        )

    if fromtype.bitwidth < totype.bitwidth:
        if fromtype.signed:
            return builder.sext(val, context.get_value_type(totype))
        else:
            return builder.zext(val, context.get_value_type(totype))
    elif fromtype.bitwidth > totype.bitwidth:
        return builder.trunc(val, context.get_value_type(totype))
    else:
        return val
