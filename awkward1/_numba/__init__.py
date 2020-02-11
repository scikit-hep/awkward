# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

try:
    import numba
except ImportError:
    installed = False
else:
    installed = True
    import awkward1._numba.cpu
    import awkward1._numba.libawkward
    import awkward1._numba.util
    import awkward1._numba.identities
    import awkward1._numba.types
    import awkward1._numba.content
    import awkward1._numba.iterator
    import awkward1._numba.fillable
    import awkward1._numba.array.numpyarray
    import awkward1._numba.array.listarray
    import awkward1._numba.array.listoffsetarray
    import awkward1._numba.array.emptyarray
    import awkward1._numba.array.regulararray
    import awkward1._numba.array.recordarray
    import awkward1._numba.array.indexedarray
