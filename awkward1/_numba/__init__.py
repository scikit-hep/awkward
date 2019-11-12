# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

try:
    import numba
except ImportError:
    installed = False
else:
    installed = True
    import awkward1._numba.cpu
    import awkward1._numba.util
    import awkward1._numba.identity
    import awkward1._numba.content
    import awkward1._numba.iterator
    import awkward1._numba.fillable
    import awkward1._numba.array.numpyarray
    import awkward1._numba.array.listarray
    import awkward1._numba.array.listoffsetarray
    import awkward1._numba.array.empty
