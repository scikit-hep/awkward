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

def test_unbox():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    @numba.njit
    def f1(x):
        return 3.14

    assert f1(array) == 3.14

def test_box():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    @numba.njit
    def f1(x):
        return x

    assert awkward1.tolist(f1(array)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_refcount():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array.numbatype
    assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.postable, array._numbaview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f1(x):
            return 3.14
        
        for j in range(10):
            f1(array)
            assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.postable, array._numbaview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f2(x):
            return x
        
        for j in range(10):
            y = f2(array)
            assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.postable, array._numbaview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f3(x):
            return x, x

        for j in range(10):
            y = f3(array)
            assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.postable, array._numbaview.lookup.arrayptrs)]

def test_len():
    array = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    @numba.njit
    def f1(x):
        return len(x)

    assert f1(array) == 5

def test_NumpyArray_getitem():
    array = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    @numba.njit
    def f1(x, i):
        return x[i]

    assert f1(array, 0) == 1.1
    assert f1(array, 1) == 2.2
    assert f1(array, 2) == 3.3
    assert f1(array, 3) == 4.4
    assert f1(array, 4) == 5.5
    assert f1(array, -1) == 5.5
    assert f1(array, -2) == 4.4
    assert f1(array, -3) == 3.3
    assert f1(array, -4) == 2.2
    assert f1(array, -5) == 1.1

    with pytest.raises(ValueError) as err:
        assert f1(array, 5)
    assert str(err.value) == "slice index out of bounds"

    with pytest.raises(ValueError) as err:
        assert f1(array, -6)
    assert str(err.value) == "slice index out of bounds"

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert awkward1.tolist(f2(array,  0, 5)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array,  1, 5)) == [     2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array,  2, 5)) == [          3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array,  3, 5)) == [               4.4, 5.5]
    assert awkward1.tolist(f2(array,  4, 5)) == [                    5.5]
    assert awkward1.tolist(f2(array,  5, 5)) == [                       ]
    assert awkward1.tolist(f2(array,  6, 5)) == [                       ]
    assert awkward1.tolist(f2(array, -1, 5)) == [                    5.5]
    assert awkward1.tolist(f2(array, -2, 5)) == [               4.4, 5.5]
    assert awkward1.tolist(f2(array, -3, 5)) == [          3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, -4, 5)) == [     2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, -5, 5)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, -6, 5)) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert awkward1.tolist(f2(array, 0, -6)) == [                       ]
    assert awkward1.tolist(f2(array, 0, -5)) == [                       ]
    assert awkward1.tolist(f2(array, 0, -4)) == [1.1                    ]
    assert awkward1.tolist(f2(array, 0, -3)) == [1.1, 2.2               ]
    assert awkward1.tolist(f2(array, 0, -2)) == [1.1, 2.2, 3.3          ]
    assert awkward1.tolist(f2(array, 0, -1)) == [1.1, 2.2, 3.3, 4.4     ]
    assert awkward1.tolist(f2(array, 0,  6)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, 0,  5)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, 0,  4)) == [1.1, 2.2, 3.3, 4.4     ]
    assert awkward1.tolist(f2(array, 0,  3)) == [1.1, 2.2, 3.3          ]
    assert awkward1.tolist(f2(array, 0,  2)) == [1.1, 2.2               ]
    assert awkward1.tolist(f2(array, 0,  1)) == [1.1                    ]
    assert awkward1.tolist(f2(array, 0,  0)) == [                       ]

    aslist = [1.1, 2.2, 3.3, 4.4, 5.5]
    for i1 in range(-6, 7):
        for i2 in range(-6, 7):
            assert awkward1.tolist(f2(array, i1, i2)) == aslist[i1:i2]

def test_RegularArray_getitem():
    array = awkward1.Array(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))

def test_ListArray_getitem():
    array = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])

def test_IndexedArray_getitem():
    content = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).layout
    index = awkward1.layout.Index64(numpy.array([3, 2, 2, 5, 0, 7], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.IndexedArray64(index, content))

def test_IndexedOptionArray_getitem():
    array = awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])

def test_RecordArray_getitem():
    array = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}])

    array = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}])
    
def test_UnionArray_getitem():
    array = awkward1.Array([1, 2, 3, [], [1], [2, 2]])

    content1 = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}]).layout
    content2 = awkward1.Array([{"y": [], "z": 0}, {"y": [1], "z": 1}, {"y": [2, 2], "z": 2}, {"y": [3, 3, 3], "z": 3}, {"y": [4, 4, 4, 4], "z": 4}]).layout
    tags  = awkward1.layout.Index8( numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content1, content2]))
