# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel
import awkward1.layout
import numpy

def register():
    import awkward1._numba.arrayview
    import awkward1._numba.layout

def repr_behavior(behavior):
    return repr(behavior)

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

    @numba.extending.typeof_impl.register(awkward1.highlevel.FillableArray)
    def typeof_FillableArray(obj, c):
        return obj.numbatype
