# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

class AwkwardLookup:
    def __init__(self, arrays):
        self.lookup = numpy.arange(len(arrays), dtype=numpy.intp)
        self.arrays = arrays

    def array(self, i):
        return self.arrays[i]

@numba.extending.typeof_impl.register(AwkwardLookup)
def typeof(obj, c):
    return AwkwardLookupType(numba.typeof(obj.lookup), tuple(numba.typeof(x) for x in obj.arrays))

class AwkwardLookupType(numba.types.Type):
    def __init__(self, lookuptype, arraytypes):
        super(AwkwardLookupType, self).__init__(name="AwkwardLookupType({0}, ({1}{2}))".format(lookuptype.name, ", ".join(x.name for x in arraytypes), "," if len(arraytypes) == 1 else ""))
        self.lookuptype = lookuptype
        self.arraytypes = arraytypes

@numba.extending.register_model(AwkwardLookupType)
class AwkwardLookupModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("lookup", fe_type.lookuptype)]
        for i, x in enumerate(fe_type.arraytypes):
            members.append(("array" + str(i), x))
        super(AwkwardLookupModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(AwkwardLookupType)
def unbox_AwkwardLookupType(altype, alobj, c):
    proxyout = numba.cgutils.create_struct_proxy(altype)(c.context, c.builder)

    lookup_obj = c.pyapi.object_getattr_string(alobj, "lookup")
    proxyout.lookup = c.pyapi.to_native_value(altype.lookuptype, lookup_obj).value
    c.pyapi.decref(lookup_obj)

    for i, arraytype in enumerate(altype.arraytypes):
        i_obj = c.pyapi.long_from_long(c.context.get_constant(numba.int64, i))
        array_obj = c.pyapi.call_method(alobj, "array", (i_obj,))
        array_val = c.pyapi.to_native_value(arraytype, array_obj).value
        setattr(proxyout, "array" + str(i), array_val)
        c.pyapi.decref(i_obj)
        c.pyapi.decref(array_obj)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)
    
def test_tests():
    awkwardlookup = AwkwardLookup([numpy.array([1, 2, 3, 4, 5]), numpy.array([1.1, 2.2, 3.3])])

    @numba.njit
    def f1(x):
        return 3.14

    f1(awkwardlookup)
