# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# from __future__ import absolute_import

# import distutils.version
# import types

# import awkward as ak

# checked_version = False


# def register_and_check():
#     global checked_version
#     try:
#         import numba
#     except ImportError:
#         raise ImportError(
#             """install the 'numba' package with:

#     pip install numba --upgrade

# or

#     conda install numba"""
#         )
#     else:
#         if not checked_version and distutils.version.LooseVersion(
#             numba.__version__
#         ) < distutils.version.LooseVersion("0.50"):
#             raise ImportError(
#                 "Awkward Array can only work with numba 0.50 or later "
#                 "(you have version {0})".format(numba.__version__)
#             )
#         checked_version = True
#         register()


# def register():
#     import numba
#     import awkward._v2._connect.numba.arrayview
#     import awkward._v2._connect.numba.layout
#     import awkward._v2._connect.numba.builder

#     if hasattr(ak.numba, "ArrayViewType"):
#         return

#     n = ak.numba
#     n.ArrayViewType = awkward._v2._connect.numba.arrayview.ArrayViewType
#     n.ArrayViewModel = awkward._v2._connect.numba.arrayview.ArrayViewModel
#     n.RecordViewType = awkward._v2._connect.numba.arrayview.RecordViewType
#     n.RecordViewModel = awkward._v2._connect.numba.arrayview.RecordViewModel
#     n.ContentType = awkward._v2._connect.numba.layout.ContentType
#     n.NumpyArrayType = awkward._v2._connect.numba.layout.NumpyArrayType
#     n.RegularArrayType = awkward._v2._connect.numba.layout.RegularArrayType
#     n.ListArrayType = awkward._v2._connect.numba.layout.ListArrayType
#     n.IndexedArrayType = awkward._v2._connect.numba.layout.IndexedArrayType
#     n.IndexedOptionArrayType = awkward._v2._connect.numba.layout.IndexedOptionArrayType
#     n.ByteMaskedArrayType = awkward._v2._connect.numba.layout.ByteMaskedArrayType
#     n.BitMaskedArrayType = awkward._v2._connect.numba.layout.BitMaskedArrayType
#     n.UnmaskedArrayType = awkward._v2._connect.numba.layout.UnmaskedArrayType
#     n.RecordArrayType = awkward._v2._connect.numba.layout.RecordArrayType
#     n.UnionArrayType = awkward._v2._connect.numba.layout.UnionArrayType
#     n.ArrayBuilderType = awkward._v2._connect.numba.builder.ArrayBuilderType
#     n.ArrayBuilderModel = awkward._v2._connect.numba.builder.ArrayBuilderModel

#     @numba.extending.typeof_impl.register(ak._v2.highlevel.Array)
#     def typeof_Array(obj, c):
#         return obj.numba_type

#     @numba.extending.typeof_impl.register(ak._v2.highlevel.Record)
#     def typeof_Record(obj, c):
#         return obj.numba_type

#     @numba.extending.typeof_impl.register(ak._v2.highlevel.ArrayBuilder)
#     def typeof_ArrayBuilder(obj, c):
#         return obj.numba_type


# def repr_behavior(behavior):
#     return repr(behavior)


# def castint(context, builder, fromtype, totype, val):
#     import numba
#     import llvmlite.ir.types

#     if isinstance(fromtype, llvmlite.ir.types.IntType):
#         if fromtype.width == 8:
#             fromtype = numba.int8
#         elif fromtype.width == 16:
#             fromtype = numba.int16
#         elif fromtype.width == 32:
#             fromtype = numba.int32
#         elif fromtype.width == 64:
#             fromtype = numba.int64
#     if not isinstance(fromtype, numba.types.Integer):
#         raise AssertionError(
#             "unrecognized integer type: {0}".format(repr(fromtype))
#
#         )

#     if fromtype.bitwidth < totype.bitwidth:
#         if fromtype.signed:
#             return builder.sext(val, context.get_value_type(totype))
#         else:
#             return builder.zext(val, context.get_value_type(totype))
#     elif fromtype.bitwidth > totype.bitwidth:
#         return builder.trunc(val, context.get_value_type(totype))
#     else:
#         return val


# ak.numba = types.ModuleType("numba")
# ak.numba.register = register
