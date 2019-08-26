# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

try:
    import numba
except ImportError:
    installed = False
else:
    installed = True
    import awkward1._numba.cpu
    import awkward1._numba.common
    import awkward1._numba.numpyarray
    import awkward1._numba.listoffsetarray
