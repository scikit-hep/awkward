# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel
import awkward1.layout
import numpy

def register():
    import awkward1._numba.layout
    import awkward1._numba.lookupview

try:
    import numba
except ImportError:
    pass
else:
    pass
