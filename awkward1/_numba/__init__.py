# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel
import awkward1.layout
import numpy

def register():
    import awkward1._numba.content
    import awkward1._numba.array.numpyarray
    import awkward1._numba.highlevel

try:
    import numba
except ImportError:
    pass
else:

    @numba.extending.typeof_impl.register(awkward1.highlevel.Array)
    def typeof_Array(val, c):
        import awkward1._numba.highlevel
        return awkward1._numba.highlevel.ArrayType(numba.typeof(val.layout), awkward1._numba.util.dict2items(val.behavior))

    @numba.extending.typeof_impl.register(awkward1.layout.NumpyArray)
    def typeof_NumpyArray(val, c):
        import awkward1._numba.array.numpyarray
        return awkward1._numba.array.numpyarray.NumpyArrayType(numba.typeof(numpy.asarray(val)), numba.typeof(val.identities), awkward1._numba.util.dict2items(val.parameters))
