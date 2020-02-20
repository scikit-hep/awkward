# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1
import awkward1._numba.arrayview

numba = pytest.importorskip("numba")

# class AwkwardLookup:
#     def __init__(self, arrays):
#         self.lookup = numpy.arange(len(arrays), dtype=numpy.intp)
#         self.arrays = arrays

#     def array(self, i):
#         return self.arrays[i]

# @numba.extending.typeof_impl.register(AwkwardLookup)
# def typeof(obj, c):
#     return AwkwardLookupType(numba.typeof(obj.lookup), tuple(numba.typeof(x) for x in obj.arrays))

# class AwkwardLookupType(numba.types.Type):
#     def __init__(self, lookuptype, arraytypes):
#         super(AwkwardLookupType, self).__init__(name="AwkwardLookupType({0}, ({1}{2}))".format(lookuptype.name, ", ".join(x.name for x in arraytypes), "," if len(arraytypes) == 1 else ""))
#         self.lookuptype = lookuptype
#         self.arraytypes = arraytypes

# @numba.extending.register_model(AwkwardLookupType)
# class AwkwardLookupModel(numba.datamodel.models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [("lookup", fe_type.lookuptype)]
#         for i, x in enumerate(fe_type.arraytypes):
#             members.append(("array" + str(i), x))
#         super(AwkwardLookupModel, self).__init__(dmm, fe_type, members)

# @numba.extending.unbox(AwkwardLookupType)
# def unbox_AwkwardLookupType(altype, alobj, c):
#     proxyout = c.context.make_helper(c.builder, altype)

#     lookup_obj = c.pyapi.object_getattr_string(alobj, "lookup")
#     proxyout.lookup = c.pyapi.to_native_value(altype.lookuptype, lookup_obj).value
#     c.pyapi.decref(lookup_obj)

#     for i, arraytype in enumerate(altype.arraytypes):
#         i_obj = c.pyapi.long_from_long(c.context.get_constant(numba.intp, i))
#         array_obj = c.pyapi.call_method(alobj, "array", (i_obj,))
#         array_val = c.pyapi.to_native_value(arraytype, array_obj).value
#         setattr(proxyout, "array" + str(i), array_val)
#         c.pyapi.decref(i_obj)
#         c.pyapi.decref(array_obj)

#     is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#     return numba.extending.NativeValue(proxyout._getvalue(), is_error)

# @numba.typing.templates.infer_getattr
# class AwkwardLookupType_methods(numba.typing.templates.AttributeTemplate):
#     key = AwkwardLookupType

#     def generic_resolve(self, altype, attr):
#         if attr == "getview":
#             return AwkwardViewType(altype, 0)

# @numba.extending.lower_getattr(AwkwardLookupType, "getview")
# def AwkwardLookupType_getview(context, builder, altype, alval):
#     alproxy = context.make_helper(builder, altype, alval)
#     arrayproxy = context.make_helper(builder, altype.arraytypes[0], getattr(alproxy, "array" + str(0)))
#     proxyout = context.make_helper(builder, AwkwardViewType(altype, 0))
#     proxyout.ptr = arrayproxy.data
#     return proxyout._getvalue()

# class AwkwardViewType(numba.types.Type):
#     def __init__(self, altype, which):
#         super(AwkwardViewType, self).__init__(name="AwkwardViewType({0}, {1})".format(altype.name, which))
#         self.altype = altype
#         self.which = which

# @numba.extending.register_model(AwkwardViewType)
# class AwkwardViewModel(numba.datamodel.models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [("ptr", numba.types.CPointer(fe_type.altype.arraytypes[fe_type.which].dtype))]
#         super(AwkwardViewModel, self).__init__(dmm, fe_type, members)

# @numba.typing.templates.infer_getattr
# class AwkwardViewType_methods(numba.typing.templates.AttributeTemplate):
#     key = AwkwardViewType

#     def generic_resolve(self, avtype, attr):
#         if attr == "getval":
#             return avtype.altype.arraytypes[avtype.which].dtype

# @numba.extending.lower_getattr(AwkwardViewType, "getval")
# def AwkwardViewType_getval(context, builder, avtype, avval):
#     avproxy = context.make_helper(builder, avtype, avval)
#     return builder.load(avproxy.ptr)

# def test_tests():
#     awkwardlookup = AwkwardLookup([numpy.array([999, 2, 3, 4, 5]), numpy.array([1.1, 2.2, 3.3])])

#     @numba.njit
#     def f1(x):
#         y = x.getview
#         return y.getval

#     assert f1(awkwardlookup) == 999

def test_views():
    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])).toarray()) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))).toarray()) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])).toarray()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([1.1, 2.2, None, 3.3, None, 4.4, 5.5])).toarray()) == [1.1, 2.2, None, 3.3, None, 4.4, 5.5]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1, 1]}, {"x": 2.2, "y": [2, 2, 2]}])).toarray()) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1, 1]}, {"x": 2.2, "y": [2, 2, 2]}]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([(0.0, []), (1.1, [1, 1]), (2.2, [2, 2, 2])])).toarray()) == [(0.0, []), (1.1, [1, 1]), (2.2, [2, 2, 2])]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([1.1, 2.2, 3.3, [], [1], [2, 2]])).toarray()) == [1.1, 2.2, 3.3, [], [1], [2, 2]]

# def test_unbox():
#     array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

#     @numba.njit
#     def f1(x):
#         return 3.14

#     f1(array)
