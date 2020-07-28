import math
import ctypes


class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]
lib = ctypes.CDLL('/home/reik/awkward-1.0/localbuild/libawkward-cpu-kernels.so')

def test_awkward_new_Identities32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    length = 3
    funcC = getattr(lib, 'awkward_new_Identities32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64)
    ret_pass = funcC(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_new_Identities64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    length = 3
    funcC = getattr(lib, 'awkward_new_Identities64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_to_Identities64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    length = 3
    width = 3
    funcC = getattr(lib, 'awkward_Identities32_to_Identities64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, length, width)
    outtoptr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_ListOffsetArray64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_ListOffsetArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_ListOffsetArray64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_ListOffsetArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_ListOffsetArray32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_ListOffsetArray32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_ListOffsetArrayU32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_ListOffsetArrayU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_ListOffsetArrayU32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_ListOffsetArrayU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_ListOffsetArray32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_ListOffsetArray32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_ListArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_ListArrayU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_ListArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_ListArrayU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_ListArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_ListArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_ListArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_ListArray32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_ListArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_ListArray32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_ListArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_ListArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_RegularArray_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 0
    size = 3
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_RegularArray')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth)
    outtoptr = [1, 2, 3, 0.0, 1, 2, 3, 1.0, 1, 2, 3, 2.0, 4, 5, 6, 0.0, 4, 5, 6, 1.0, 4, 5, 6, 2.0, 7, 8, 9, 0.0, 7, 8, 9, 1.0, 7, 8, 9, 2.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_RegularArray_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 0
    size = 3
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_RegularArray')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth)
    outtoptr = [1, 2, 3, 0.0, 1, 2, 3, 1.0, 1, 2, 3, 2.0, 4, 5, 6, 0.0, 4, 5, 6, 1.0, 4, 5, 6, 2.0, 7, 8, 9, 0.0, 7, 8, 9, 1.0, 7, 8, 9, 2.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_IndexedArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_IndexedArray32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_IndexedArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_IndexedArrayU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_IndexedArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities64_from_IndexedArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_IndexedArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_IndexedArray32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_IndexedArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_IndexedArrayU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_IndexedArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcC = getattr(lib, 'awkward_Identities32_from_IndexedArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_UnionArray8_32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcC = getattr(lib, 'awkward_Identities64_from_UnionArray8_32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_UnionArray8_64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcC = getattr(lib, 'awkward_Identities32_from_UnionArray8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_UnionArray8_32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcC = getattr(lib, 'awkward_Identities32_from_UnionArray8_32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_UnionArray8_64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcC = getattr(lib, 'awkward_Identities64_from_UnionArray8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_from_UnionArray8_U32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcC = getattr(lib, 'awkward_Identities64_from_UnionArray8_U32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_from_UnionArray8_U32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    uniquecontents = (ctypes.c_bool*len(uniquecontents))(*uniquecontents)
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcC = getattr(lib, 'awkward_Identities32_from_UnionArray8_U32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert math.isclose(uniquecontents[i], outuniquecontents[i], rel_tol=0.0001)
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_extend_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromoffset = 0
    fromlength = 3
    tolength = 3
    funcC = getattr(lib, 'awkward_Identities32_extend')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffset, fromlength, tolength)
    outtoptr = [1, 2, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_extend_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromoffset = 0
    fromlength = 3
    tolength = 3
    funcC = getattr(lib, 'awkward_Identities64_extend')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromoffset, fromlength, tolength)
    outtoptr = [1, 2, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tonum = (ctypes.c_int64*len(tonum))(*tonum)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_num_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length)
    outtonum = [1.0, 1.0, 1.0]
    for i in range(len(outtonum)):
        assert math.isclose(tonum[i], outtonum[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tonum = (ctypes.c_int64*len(tonum))(*tonum)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArray32_num_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length)
    outtonum = [1.0, 1.0, 1.0]
    for i in range(len(outtonum)):
        assert math.isclose(tonum[i], outtonum[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tonum = (ctypes.c_int64*len(tonum))(*tonum)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArray64_num_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length)
    outtonum = [1.0, 1.0, 1.0]
    for i in range(len(outtonum)):
        assert math.isclose(tonum[i], outtonum[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tonum = (ctypes.c_int64*len(tonum))(*tonum)
    size = 3
    length = 3
    funcC = getattr(lib, 'awkward_RegularArray_num_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tonum, size, length)
    outtonum = [3, 3, 3]
    for i in range(len(outtonum)):
        assert math.isclose(tonum[i], outtonum[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray64_flatten_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    outeroffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    outeroffsets = (ctypes.c_int64*len(outeroffsets))(*outeroffsets)
    outeroffsetsoffset = 1
    outeroffsetslen = 3
    inneroffsets = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    inneroffsets = (ctypes.c_int64*len(inneroffsets))(*inneroffsets)
    inneroffsetsoffset = 0
    inneroffsetslen = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray64_flatten_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen)
    outtooffsets = [50, 51, 55]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArrayU32_flatten_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    outeroffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    outeroffsets = (ctypes.c_uint32*len(outeroffsets))(*outeroffsets)
    outeroffsetsoffset = 1
    outeroffsetslen = 3
    inneroffsets = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    inneroffsets = (ctypes.c_int64*len(inneroffsets))(*inneroffsets)
    inneroffsetsoffset = 0
    inneroffsetslen = 3
    funcC = getattr(lib, 'awkward_ListOffsetArrayU32_flatten_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen)
    outtooffsets = [50, 51, 55]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray32_flatten_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    outeroffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    outeroffsets = (ctypes.c_int32*len(outeroffsets))(*outeroffsets)
    outeroffsetsoffset = 1
    outeroffsetslen = 3
    inneroffsets = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    inneroffsets = (ctypes.c_int64*len(inneroffsets))(*inneroffsets)
    inneroffsetsoffset = 0
    inneroffsetslen = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray32_flatten_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen)
    outtooffsets = [50, 51, 55]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_flatten_none2empty_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outoffsets = (ctypes.c_int64*len(outoffsets))(*outoffsets)
    outindex = [1, 0, 0, 1, 1, 1, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    outindexoffset = 1
    outindexlength = 3
    offsets = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_flatten_none2empty_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength)
    outoutoffsets = [14, 1, -12, 14]
    for i in range(len(outoutoffsets)):
        assert math.isclose(outoffsets[i], outoutoffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_flatten_none2empty_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outoffsets = (ctypes.c_int64*len(outoffsets))(*outoffsets)
    outindex = [1, 0, 0, 1, 1, 1, 0]
    outindex = (ctypes.c_int32*len(outindex))(*outindex)
    outindexoffset = 1
    outindexlength = 3
    offsets = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_flatten_none2empty_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength)
    outoutoffsets = [14, 1, -12, 14]
    for i in range(len(outoutoffsets)):
        assert math.isclose(outoffsets[i], outoutoffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_flatten_none2empty_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outoffsets = (ctypes.c_int64*len(outoffsets))(*outoffsets)
    outindex = [1, 0, 0, 1, 1, 1, 0]
    outindex = (ctypes.c_uint32*len(outindex))(*outindex)
    outindexoffset = 1
    outindexlength = 3
    offsets = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_flatten_none2empty_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength)
    outoutoffsets = [14, 1, -12, 14]
    for i in range(len(outoutoffsets)):
        assert math.isclose(outoffsets[i], outoutoffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray64_flatten_length_64_1():
    total_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_length = (ctypes.c_int64*len(total_length))(*total_length)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsraws = ctypes.pointer(ctypes.cast((ctypes.c_int64*len(offsetsraws[0]))(*offsetsraws[0]),ctypes.POINTER(ctypes.c_int64)))
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    offsetsoffsets = (ctypes.c_int64*len(offsetsoffsets))(*offsetsoffsets)
    funcC = getattr(lib, 'awkward_UnionArray64_flatten_length_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotal_length = [1]
    for i in range(len(outtotal_length)):
        assert math.isclose(total_length[i], outtotal_length[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArrayU32_flatten_length_64_1():
    total_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_length = (ctypes.c_int64*len(total_length))(*total_length)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsraws = ctypes.pointer(ctypes.cast((ctypes.c_int64*len(offsetsraws[0]))(*offsetsraws[0]),ctypes.POINTER(ctypes.c_int64)))
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    offsetsoffsets = (ctypes.c_int64*len(offsetsoffsets))(*offsetsoffsets)
    funcC = getattr(lib, 'awkward_UnionArrayU32_flatten_length_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotal_length = [1]
    for i in range(len(outtotal_length)):
        assert math.isclose(total_length[i], outtotal_length[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray32_flatten_length_64_1():
    total_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_length = (ctypes.c_int64*len(total_length))(*total_length)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsraws = ctypes.pointer(ctypes.cast((ctypes.c_int64*len(offsetsraws[0]))(*offsetsraws[0]),ctypes.POINTER(ctypes.c_int64)))
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    offsetsoffsets = (ctypes.c_int64*len(offsetsoffsets))(*offsetsoffsets)
    funcC = getattr(lib, 'awkward_UnionArray32_flatten_length_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotal_length = [1]
    for i in range(len(outtotal_length)):
        assert math.isclose(total_length[i], outtotal_length[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray32_flatten_combine_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsraws = ctypes.pointer(ctypes.cast((ctypes.c_int64*len(offsetsraws[0]))(*offsetsraws[0]),ctypes.POINTER(ctypes.c_int64)))
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    offsetsoffsets = (ctypes.c_int64*len(offsetsoffsets))(*offsetsoffsets)
    funcC = getattr(lib, 'awkward_UnionArray32_flatten_combine_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotags = [0, 0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [1, 3]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtooffsets = [0, 1, 0, 1]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArrayU32_flatten_combine_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsraws = ctypes.pointer(ctypes.cast((ctypes.c_int64*len(offsetsraws[0]))(*offsetsraws[0]),ctypes.POINTER(ctypes.c_int64)))
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    offsetsoffsets = (ctypes.c_int64*len(offsetsoffsets))(*offsetsoffsets)
    funcC = getattr(lib, 'awkward_UnionArrayU32_flatten_combine_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotags = [0, 0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [1, 3]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtooffsets = [0, 1, 0, 1]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray64_flatten_combine_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsraws = ctypes.pointer(ctypes.cast((ctypes.c_int64*len(offsetsraws[0]))(*offsetsraws[0]),ctypes.POINTER(ctypes.c_int64)))
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    offsetsoffsets = (ctypes.c_int64*len(offsetsoffsets))(*offsetsoffsets)
    funcC = getattr(lib, 'awkward_UnionArray64_flatten_combine_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotags = [0, 0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [1, 3]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtooffsets = [0, 1, 0, 1]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_flatten_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_flatten_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_flatten_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_flatten_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_flatten_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_flatten_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_overlay_mask8_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_overlay_mask8_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, mask, maskoffset, fromindex, indexoffset, length)
    outtoindex = [0, 0, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_overlay_mask8_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_overlay_mask8_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, mask, maskoffset, fromindex, indexoffset, length)
    outtoindex = [0, 0, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_overlay_mask8_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_overlay_mask8_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, mask, maskoffset, fromindex, indexoffset, length)
    outtoindex = [0, 0, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_mask8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomask, fromindex, indexoffset, length)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_mask8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomask, fromindex, indexoffset, length)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_mask8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomask, fromindex, indexoffset, length)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    frommask = [1, 1, 1, 1, 1]
    frommask = (ctypes.c_int8*len(frommask))(*frommask)
    maskoffset = 0
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_mask8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(tomask, frommask, maskoffset, length, validwhen)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_zero_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    length = 3
    funcC = getattr(lib, 'awkward_zero_mask8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64)
    ret_pass = funcC(tomask, length)
    outtomask = [0, 0, 0]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_simplifyU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_int32*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_uint32*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_simplifyU32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_simplify64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_uint32*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_int64*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_simplify64_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_simplify64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_int64*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_int64*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_simplify64_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_simplifyU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_uint32*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_uint32*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_simplifyU32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_simplify64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_int32*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_int64*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_simplify64_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_simplify32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_int64*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_int32*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_simplify32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_simplify32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_int32*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_int32*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_simplify32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_simplify32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_uint32*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_int32*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_simplify32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_simplifyU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outerindex = (ctypes.c_int64*len(outerindex))(*outerindex)
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    innerindex = (ctypes.c_uint32*len(innerindex))(*innerindex)
    inneroffset = 0
    innerlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_simplifyU32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_compact_offsets64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    length = 3
    size = 3
    funcC = getattr(lib, 'awkward_RegularArray_compact_offsets64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, length, size)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    startsoffset = 0
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArray32_compact_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    startsoffset = 0
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_compact_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    startsoffset = 0
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArray64_compact_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray64_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray64_compact_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, fromoffsets, offsetsoffset, length)
    outtooffsets = [0, 1, 2, 4]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray32_compact_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, fromoffsets, offsetsoffset, length)
    outtooffsets = [0, 1, 2, 4]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArrayU32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListOffsetArrayU32_compact_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, fromoffsets, offsetsoffset, length)
    outtooffsets = [0, 1, 2, 4]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_broadcast_tooffsets_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromoffsets = [1, 2, 3, 4, 5, 6]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    stopsoffset = 0
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_broadcast_tooffsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent)
    outtocarry = [2.0, 0.0]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_broadcast_tooffsets_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromoffsets = [1, 2, 3, 4, 5, 6]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    stopsoffset = 0
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArray64_broadcast_tooffsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent)
    outtocarry = [2.0, 0.0]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_broadcast_tooffsets_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromoffsets = [1, 2, 3, 4, 5, 6]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    stopsoffset = 0
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArray32_broadcast_tooffsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent)
    outtocarry = [2.0, 0.0]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_broadcast_tooffsets_64_1():
    fromoffsets = [0, 3, 2, 9, 12, 15]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    size = 2
    funcC = getattr(lib, 'awkward_RegularArray_broadcast_tooffsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    assert funcC(fromoffsets, offsetsoffset, offsetslength, size).str.contents

def test_awkward_RegularArray_broadcast_tooffsets_size1_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromoffsets = [0, 3, 6, 9, 12, 15]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_RegularArray_broadcast_tooffsets_size1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromoffsets, offsetsoffset, offsetslength)
    outtocarry = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray64_toRegularArray_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = (ctypes.c_int64*len(size))(*size)
    fromoffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray64_toRegularArray')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(size, fromoffsets, offsetsoffset, offsetslength)
    outsize = [1]
    for i in range(len(outsize)):
        assert math.isclose(size[i], outsize[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArrayU32_toRegularArray_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = (ctypes.c_int64*len(size))(*size)
    fromoffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArrayU32_toRegularArray')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(size, fromoffsets, offsetsoffset, offsetslength)
    outsize = [1]
    for i in range(len(outsize)):
        assert math.isclose(size[i], outsize[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray32_toRegularArray_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = (ctypes.c_int64*len(size))(*size)
    fromoffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 0
    offsetslength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray32_toRegularArray')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(size, fromoffsets, offsetsoffset, offsetslength)
    outsize = [1]
    for i in range(len(outsize)):
        assert math.isclose(size[i], outsize[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint32_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint32_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint16_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint16_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint8_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int8*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint8_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint16_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint16_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint64_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint64_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint8_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint8*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint8_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint32_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint32_frombool')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint32_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint32_fromuint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint8_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint8*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint8_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromuint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromuint32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint64_fromuint64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint64_fromuint64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint32_fromint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint32_fromint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint64_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint64_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint32_fromuint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint32_fromuint32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint16_fromint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint16_fromint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint8_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int8*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint8_fromint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint64_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint64_fromuint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint16_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint16_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint32_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint32_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromint32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint16_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint16_fromint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint32_fromint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint32_fromint32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromuint64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromuint64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromuint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint32_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint32_fromuint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint32_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint32_fromint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint16_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint16_fromuint16')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint32_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint32_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_toint64_fromint64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_toint64_fromint64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint64_fromuint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint64_fromuint32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_fill_touint16_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint16*len(toptr))(*toptr)
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_NumpyArray_fill_touint16_fromuint8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray_fill_to64_fromU32_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int64*len(tostarts))(*tostarts)
    tostartsoffset = 3
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int64*len(tostops))(*tostops)
    tostopsoffset = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_ListArray_fill_to64_fromU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base)
    outtostarts = [0, 0, 0, 5.0, 3.0, 5.0]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [0, 0, 0, 6.0, 4.0, 6.0]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray_fill_to64_from32_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int64*len(tostarts))(*tostarts)
    tostartsoffset = 3
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int64*len(tostops))(*tostops)
    tostopsoffset = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_ListArray_fill_to64_from32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base)
    outtostarts = [0, 0, 0, 5.0, 3.0, 5.0]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [0, 0, 0, 6.0, 4.0, 6.0]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray_fill_to64_from64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int64*len(tostarts))(*tostarts)
    tostartsoffset = 3
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int64*len(tostops))(*tostops)
    tostopsoffset = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_ListArray_fill_to64_from64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base)
    outtostarts = [0, 0, 0, 5.0, 3.0, 5.0]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [0, 0, 0, 6.0, 4.0, 6.0]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray_fill_to64_from32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromindexoffset = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_IndexedArray_fill_to64_from32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, fromindex, fromindexoffset, length, base)
    outtoindex = [0, 0, 0, 3.0, 3.0, 4.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray_fill_to64_fromU32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromindexoffset = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_IndexedArray_fill_to64_fromU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, fromindex, fromindexoffset, length, base)
    outtoindex = [0, 0, 0, 3.0, 3.0, 4.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray_fill_to64_from64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromindexoffset = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_IndexedArray_fill_to64_from64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, fromindex, fromindexoffset, length, base)
    outtoindex = [0, 0, 0, 3.0, 3.0, 4.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray_fill_to64_count_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_IndexedArray_fill_to64_count')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, length, base)
    outtoindex = [0, 3, 4, 5]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_filltags_to8_from8_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    totagsoffset = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray_filltags_to8_from8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, totagsoffset, fromtags, fromtagsoffset, length, base)
    outtotags = [0, 0, 0, 3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_fillindex_to64_from64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillindex_to64_from64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, fromindex, fromindexoffset, length)
    outtoindex = [0, 0, 0, 4.0, 3.0, 6.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_fillindex_to64_from32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillindex_to64_from32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, fromindex, fromindexoffset, length)
    outtoindex = [0, 0, 0, 4.0, 3.0, 6.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_fillindex_to64_fromU32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillindex_to64_fromU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, fromindex, fromindexoffset, length)
    outtoindex = [0, 0, 0, 4.0, 3.0, 6.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_filltags_to8_const_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    totagsoffset = 3
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray_filltags_to8_const')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, totagsoffset, length, base)
    outtotags = [0, 0, 0, 3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_fillindex_to64_count_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    toindexoffset = 3
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillindex_to64_count')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, toindexoffset, length)
    outtoindex = [0, 0, 0, 0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_U32_simplify8_32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_uint32*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_int32*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_U32_simplify8_32_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_64_simplify8_32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_int64*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_int32*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_64_simplify8_32_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_64_simplify8_64_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_int64*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_int64*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_64_simplify8_64_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_simplify8_U32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_int32*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_uint32*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_32_simplify8_U32_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_U32_simplify8_U32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_uint32*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_uint32*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_U32_simplify8_U32_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_simplify8_32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_int32*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_int32*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_32_simplify8_32_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_64_simplify8_U32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_int64*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_uint32*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_64_simplify8_U32_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_U32_simplify8_64_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_uint32*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_int64*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_U32_simplify8_64_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_simplify8_64_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = (ctypes.c_int8*len(outertags))(*outertags)
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindex = (ctypes.c_int32*len(outerindex))(*outerindex)
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertags = (ctypes.c_int8*len(innertags))(*innertags)
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindex = (ctypes.c_int64*len(innerindex))(*innerindex)
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_32_simplify8_64_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_simplify_one_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    towhich = 3
    fromwhich = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_32_simplify_one_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base)
    outtotags = []
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = []
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_U32_simplify_one_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromindexoffset = 0
    towhich = 3
    fromwhich = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_U32_simplify_one_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base)
    outtotags = []
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = []
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_64_simplify_one_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totags = (ctypes.c_int8*len(totags))(*totags)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromindexoffset = 0
    towhich = 3
    fromwhich = 1
    length = 3
    base = 3
    funcC = getattr(lib, 'awkward_UnionArray8_64_simplify_one_to8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base)
    outtotags = []
    for i in range(len(outtotags)):
        assert math.isclose(totags[i], outtotags[i], rel_tol=0.0001)
    outtoindex = []
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_validity_1():
    tags = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tags = (ctypes.c_int8*len(tags))(*tags)
    tagsoffset = 1
    index = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    index = (ctypes.c_int32*len(index))(*index)
    indexoffset = 0
    length = 3
    numcontents = 3
    lencontents = [7, 8, 9, 10, 11, 12]
    lencontents = (ctypes.c_int64*len(lencontents))(*lencontents)
    funcC = getattr(lib, 'awkward_UnionArray8_32_validity')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))
    assert funcC(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents).str.contents

def test_awkward_UnionArray8_U32_validity_1():
    tags = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tags = (ctypes.c_int8*len(tags))(*tags)
    tagsoffset = 1
    index = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    index = (ctypes.c_uint32*len(index))(*index)
    indexoffset = 0
    length = 3
    numcontents = 3
    lencontents = [7, 8, 9, 10, 11, 12]
    lencontents = (ctypes.c_int64*len(lencontents))(*lencontents)
    funcC = getattr(lib, 'awkward_UnionArray8_U32_validity')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))
    assert funcC(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents).str.contents

def test_awkward_UnionArray8_64_validity_1():
    tags = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tags = (ctypes.c_int8*len(tags))(*tags)
    tagsoffset = 1
    index = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    index = (ctypes.c_int64*len(index))(*index)
    indexoffset = 0
    length = 3
    numcontents = 3
    lencontents = [7, 8, 9, 10, 11, 12]
    lencontents = (ctypes.c_int64*len(lencontents))(*lencontents)
    funcC = getattr(lib, 'awkward_UnionArray8_64_validity')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))
    assert funcC(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents).str.contents

def test_awkward_UnionArray_fillna_from64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    offset = 0
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillna_from64_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, offset, length)
    outtoindex = [4, 3, 6]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_fillna_fromU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    offset = 0
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillna_fromU32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, offset, length)
    outtoindex = [4, 3, 6]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray_fillna_from32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    offset = 0
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray_fillna_from32_to64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, offset, length)
    outtoindex = [4, 3, 6]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    frommask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frommask = (ctypes.c_int8*len(frommask))(*frommask)
    length = 3
    funcC = getattr(lib, 'awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64)
    ret_pass = funcC(toindex, frommask, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_index_rpad_and_clip_axis0_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    target = 3
    length = 3
    funcC = getattr(lib, 'awkward_index_rpad_and_clip_axis0_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, target, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_index_rpad_and_clip_axis1_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int64*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int64*len(tostops))(*tostops)
    target = 3
    length = 3
    funcC = getattr(lib, 'awkward_index_rpad_and_clip_axis1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostops, target, length)
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_rpad_and_clip_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    target = 3
    size = 3
    length = 3
    funcC = getattr(lib, 'awkward_RegularArray_rpad_and_clip_axis1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, target, size, length)
    outtoindex = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_min_range_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomin = (ctypes.c_int64*len(tomin))(*tomin)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArray64_min_range')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset)
    outtomin = [1]
    for i in range(len(outtomin)):
        assert math.isclose(tomin[i], outtomin[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_min_range_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomin = (ctypes.c_int64*len(tomin))(*tomin)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArrayU32_min_range')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset)
    outtomin = [1]
    for i in range(len(outtomin)):
        assert math.isclose(tomin[i], outtomin[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_min_range_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomin = (ctypes.c_int64*len(tomin))(*tomin)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArray32_min_range')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset)
    outtomin = [1]
    for i in range(len(outtomin)):
        assert math.isclose(tomin[i], outtomin[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_rpad_and_clip_length_axis1_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomin = (ctypes.c_int64*len(tomin))(*tomin)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    target = 3
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArrayU32_rpad_and_clip_length_axis1')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset)
    outtomin = [9]
    for i in range(len(outtomin)):
        assert math.isclose(tomin[i], outtomin[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_rpad_and_clip_length_axis1_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomin = (ctypes.c_int64*len(tomin))(*tomin)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    target = 3
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArray32_rpad_and_clip_length_axis1')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset)
    outtomin = [9]
    for i in range(len(outtomin)):
        assert math.isclose(tomin[i], outtomin[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_rpad_and_clip_length_axis1_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomin = (ctypes.c_int64*len(tomin))(*tomin)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    target = 3
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArray64_rpad_and_clip_length_axis1')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset)
    outtomin = [9]
    for i in range(len(outtomin)):
        assert math.isclose(tomin[i], outtomin[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_rpad_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_uint32*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_uint32*len(tostops))(*tostops)
    target = 3
    length = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArrayU32_rpad_axis1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset)
    outtoindex = [2, -1, -1, 0, -1, -1, 2, -1, -1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_rpad_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int64*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int64*len(tostops))(*tostops)
    target = 3
    length = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArray64_rpad_axis1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset)
    outtoindex = [2, -1, -1, 0, -1, -1, 2, -1, -1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_rpad_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int32*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int32*len(tostops))(*tostops)
    target = 3
    length = 3
    startsoffset = 0
    stopsoffset = 0
    funcC = getattr(lib, 'awkward_ListArray32_rpad_axis1_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset)
    outtoindex = [2, -1, -1, 0, -1, -1, 2, -1, -1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArrayU32_rpad_length_axis1_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_uint32*len(tooffsets))(*tooffsets)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    fromlength = 3
    target = 3
    tolength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tolength = (ctypes.c_int64*len(tolength))(*tolength)
    funcC = getattr(lib, 'awkward_ListOffsetArrayU32_rpad_length_axis1')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtolength = [9]
    for i in range(len(outtolength)):
        assert math.isclose(tolength[i], outtolength[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray64_rpad_length_axis1_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    fromlength = 3
    target = 3
    tolength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tolength = (ctypes.c_int64*len(tolength))(*tolength)
    funcC = getattr(lib, 'awkward_ListOffsetArray64_rpad_length_axis1')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtolength = [9]
    for i in range(len(outtolength)):
        assert math.isclose(tolength[i], outtolength[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray32_rpad_length_axis1_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int32*len(tooffsets))(*tooffsets)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    fromlength = 3
    target = 3
    tolength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tolength = (ctypes.c_int64*len(tolength))(*tolength)
    funcC = getattr(lib, 'awkward_ListOffsetArray32_rpad_length_axis1')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64))
    ret_pass = funcC(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtolength = [9]
    for i in range(len(outtolength)):
        assert math.isclose(tolength[i], outtolength[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    length = 3
    funcC = getattr(lib, 'awkward_localindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(toindex, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListArray64_localindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, offsets, offsetsoffset, length)
    outtoindex = [0, 0, 0, 0, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsets = (ctypes.c_int32*len(offsets))(*offsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListArray32_localindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, offsets, offsetsoffset, length)
    outtoindex = [0, 0, 0, 0, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsets = (ctypes.c_uint32*len(offsets))(*offsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_localindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, offsets, offsetsoffset, length)
    outtoindex = [0, 0, 0, 0, 1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    size = 3
    length = 3
    funcC = getattr(lib, 'awkward_RegularArray_localindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, size, length)
    outtoindex = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_combinations_length_64_1():
    totallen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totallen = (ctypes.c_int64*len(totallen))(*totallen)
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    n = 3
    replacement = True
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    stops = [3, 1, 3, 2, 3]
    stops = (ctypes.c_int64*len(stops))(*stops)
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArray64_combinations_length_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_bool, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totallen, tooffsets, n, replacement, starts, startsoffset, stops, stopsoffset, length)
    outtotallen = [3]
    for i in range(len(outtotallen)):
        assert math.isclose(totallen[i], outtotallen[i], rel_tol=0.0001)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_combinations_length_64_1():
    totallen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totallen = (ctypes.c_int64*len(totallen))(*totallen)
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    n = 3
    replacement = True
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_uint32*len(starts))(*starts)
    startsoffset = 0
    stops = [3, 1, 3, 2, 3]
    stops = (ctypes.c_uint32*len(stops))(*stops)
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_combinations_length_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_bool, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totallen, tooffsets, n, replacement, starts, startsoffset, stops, stopsoffset, length)
    outtotallen = [3]
    for i in range(len(outtotallen)):
        assert math.isclose(totallen[i], outtotallen[i], rel_tol=0.0001)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_combinations_length_64_1():
    totallen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totallen = (ctypes.c_int64*len(totallen))(*totallen)
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    n = 3
    replacement = True
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int32*len(starts))(*starts)
    startsoffset = 0
    stops = [3, 1, 3, 2, 3]
    stops = (ctypes.c_int32*len(stops))(*stops)
    stopsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_ListArray32_combinations_length_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_bool, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(totallen, tooffsets, n, replacement, starts, startsoffset, stops, stopsoffset, length)
    outtotallen = [3]
    for i in range(len(outtotallen)):
        assert math.isclose(totallen[i], outtotallen[i], rel_tol=0.0001)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_overlay_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    theirmask = [1, 1, 1, 1, 1]
    theirmask = (ctypes.c_int8*len(theirmask))(*theirmask)
    theirmaskoffset = 0
    mymask = [1, 1, 1, 1, 1]
    mymask = (ctypes.c_int8*len(mymask))(*mymask)
    mymaskoffset = 0
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_overlay_mask8')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(tomask, theirmask, theirmaskoffset, mymask, mymaskoffset, length, validwhen)
    outtomask = [1, 1, 1]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_BitMaskedArray_to_ByteMaskedArray_1():
    tobytemask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tobytemask = (ctypes.c_int8*len(tobytemask))(*tobytemask)
    frombitmask = [244, 251, 64, 0, 133]
    frombitmask = (ctypes.c_uint8*len(frombitmask))(*frombitmask)
    bitmaskoffset = 1
    bitmasklength = 3
    validwhen = True
    lsb_order = False
    funcC = getattr(lib, 'awkward_BitMaskedArray_to_ByteMaskedArray')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool, ctypes.c_bool)
    ret_pass = funcC(tobytemask, frombitmask, bitmaskoffset, bitmasklength, validwhen, lsb_order)
    outtobytemask = [False, False, False, False, False, True, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    for i in range(len(outtobytemask)):
        assert math.isclose(tobytemask[i], outtobytemask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_BitMaskedArray_to_IndexedOptionArray64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    frombitmask = [244, 251, 64, 0, 133]
    frombitmask = (ctypes.c_uint8*len(frombitmask))(*frombitmask)
    bitmaskoffset = 1
    bitmasklength = 3
    validwhen = True
    lsb_order = False
    funcC = getattr(lib, 'awkward_BitMaskedArray_to_IndexedOptionArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool, ctypes.c_bool)
    ret_pass = funcC(toindex, frombitmask, bitmaskoffset, bitmasklength, validwhen, lsb_order)
    outtoindex = [0, 1, 2, 3, 4, -1, 6, 7, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_count_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    parents = [1, 0, 0, 1, 1, 1, 0]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 1
    lenparents = 3
    outlength = 3
    funcC = getattr(lib, 'awkward_reduce_count_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, parents, parentsoffset, lenparents, outlength)
    outtoptr = [2, 1, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_countnonzero_float32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1.1, 0.5, 1.3, 4.2, 2.1, 4.4, 1.5, 10.1, 5.6, 7.7, 1.9, 2.0, 3.0]
    fromptr = (ctypes.c_float*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_countnonzero_float32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_countnonzero_float64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1.1, 0.5, 1.3, 4.2, 2.1, 4.4, 1.5, 10.1, 5.6, 7.7, 1.9, 2.0, 3.0]
    fromptr = (ctypes.c_double*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_countnonzero_float64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint64_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint64_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint32_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint32_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int64_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int64_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int64_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint64_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint64_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint32_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint32_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int32_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int32_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint64_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint64_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int32_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int32_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint32_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_uint64_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int64_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int64_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int64_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int64_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int32_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int64_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int64_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_int32_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_int32_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_uint8_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_uint16_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_uint64_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_int32_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_uint32_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_int16_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_int8_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_sum_bool_int64_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_sum_bool_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int64_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int64_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int64_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int32_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int32_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int64_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int64_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint32_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint64_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int32_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint32_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint32_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint64_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint64_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint32_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint32_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint64_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint64_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int32_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int32_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int64_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int64_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_uint64_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_uint64_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int64_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int64_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_int32_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_int32_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_prod_bool_bool_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = (ctypes.c_bool*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_prod_bool_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_uint8_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint8*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_uint8_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint8)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_uint32_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_int8_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int8*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_int8_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int8)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_int64_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_int16_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int16*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_int16_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int16)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_int32_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_uint16_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint16*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_uint16_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint16)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_min_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_min_uint64_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_int64_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_uint16_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint16*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_uint16_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint16)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_int32_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_int16_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int16*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_int16_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int16)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_uint64_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_int8_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int8*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_int8_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int8)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_uint8_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint8*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_uint8_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint8)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_max_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcC = getattr(lib, 'awkward_reduce_max_uint32_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32)
    ret_pass = funcC(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmin_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmin_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint64*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_uint64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int64*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_int64_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_int32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_uint8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint16*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_uint16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_int8_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int16*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_int16_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_uint32_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_reduce_argmax_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptr = (ctypes.c_bool*len(fromptr))(*fromptr)
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcC = getattr(lib, 'awkward_reduce_argmax_bool_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_bool), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_content_reduce_zeroparents_64_1():
    toparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toparents = (ctypes.c_int64*len(toparents))(*toparents)
    length = 3
    funcC = getattr(lib, 'awkward_content_reduce_zeroparents_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(toparents, length)
    outtoparents = [0, 0, 0]
    for i in range(len(outtoparents)):
        assert math.isclose(toparents[i], outtoparents[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_global_startstop_64_1():
    globalstart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    globalstart = (ctypes.c_int64*len(globalstart))(*globalstart)
    globalstop = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    globalstop = (ctypes.c_int64*len(globalstop))(*globalstop)
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_global_startstop_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(globalstart, globalstop, offsets, offsetsoffset, length)
    outglobalstart = [1]
    for i in range(len(outglobalstart)):
        assert math.isclose(globalstart[i], outglobalstart[i], rel_tol=0.0001)
    outglobalstop = [5]
    for i in range(len(outglobalstop)):
        assert math.isclose(globalstop[i], outglobalstop[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_1():
    maxcount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maxcount = (ctypes.c_int64*len(maxcount))(*maxcount)
    offsetscopy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsetscopy = (ctypes.c_int64*len(offsetscopy))(*offsetscopy)
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(maxcount, offsetscopy, offsets, offsetsoffset, length)
    outmaxcount = [2]
    for i in range(len(outmaxcount)):
        assert math.isclose(maxcount[i], outmaxcount[i], rel_tol=0.0001)
    outoffsetscopy = [1, 2, 3, 5]
    for i in range(len(outoffsetscopy)):
        assert math.isclose(offsetscopy[i], outoffsetscopy[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_1():
    nextstarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextstarts = (ctypes.c_int64*len(nextstarts))(*nextstarts)
    nextparents = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    nextparents = (ctypes.c_int64*len(nextparents))(*nextparents)
    nextlen = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(nextstarts, nextparents, nextlen)
    outnextstarts = [0, 0, 2]
    for i in range(len(outnextstarts)):
        assert math.isclose(nextstarts[i], outnextstarts[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_nonlocal_findgaps_64_1():
    gaps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gaps = (ctypes.c_int64*len(gaps))(*gaps)
    parents = [1, 0, 0, 1, 1, 1, 0]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 1
    lenparents = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_nonlocal_findgaps_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(gaps, parents, parentsoffset, lenparents)
    outgaps = [1, 1]
    for i in range(len(outgaps)):
        assert math.isclose(gaps[i], outgaps[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_1():
    outstarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outstarts = (ctypes.c_int64*len(outstarts))(*outstarts)
    outstops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outstops = (ctypes.c_int64*len(outstops))(*outstops)
    distincts = [1, 0, 0, 1, 1, 1, 0]
    distincts = (ctypes.c_int64*len(distincts))(*distincts)
    lendistincts = 3
    gaps = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    gaps = (ctypes.c_int64*len(gaps))(*gaps)
    outlength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(outstarts, outstops, distincts, lendistincts, gaps, outlength)
    outoutstarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(outoutstarts)):
        assert math.isclose(outstarts[i], outoutstarts[i], rel_tol=0.0001)
    outoutstops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    for i in range(len(outoutstops)):
        assert math.isclose(outstops[i], outoutstops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_local_nextparents_64_1():
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = (ctypes.c_int64*len(nextparents))(*nextparents)
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsets = (ctypes.c_int64*len(offsets))(*offsets)
    offsetsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_local_nextparents_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextparents, offsets, offsetsoffset, length)
    outnextparents = [0, 1, 2, 2]
    for i in range(len(outnextparents)):
        assert math.isclose(nextparents[i], outnextparents[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_reduce_local_outoffsets_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outoffsets = (ctypes.c_int64*len(outoffsets))(*outoffsets)
    parents = [1, 0, 0, 1, 1, 1, 0]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 1
    lenparents = 3
    outlength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_reduce_local_outoffsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(outoffsets, parents, parentsoffset, lenparents, outlength)
    outoutoffsets = [0, 2, 3, 3]
    for i in range(len(outoutoffsets)):
        assert math.isclose(outoffsets[i], outoutoffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarry = (ctypes.c_int64*len(nextcarry))(*nextcarry)
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = (ctypes.c_int64*len(nextparents))(*nextparents)
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_uint32*len(index))(*index)
    indexoffset = 1
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_reduce_next_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length)
    outnextcarry = [0, 0, 1]
    for i in range(len(outnextcarry)):
        assert math.isclose(nextcarry[i], outnextcarry[i], rel_tol=0.0001)
    outnextparents = [1, 2, 3]
    for i in range(len(outnextparents)):
        assert math.isclose(nextparents[i], outnextparents[i], rel_tol=0.0001)
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert math.isclose(outindex[i], outoutindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarry = (ctypes.c_int64*len(nextcarry))(*nextcarry)
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = (ctypes.c_int64*len(nextparents))(*nextparents)
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_int64*len(index))(*index)
    indexoffset = 1
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_reduce_next_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length)
    outnextcarry = [0, 0, 1]
    for i in range(len(outnextcarry)):
        assert math.isclose(nextcarry[i], outnextcarry[i], rel_tol=0.0001)
    outnextparents = [1, 2, 3]
    for i in range(len(outnextparents)):
        assert math.isclose(nextparents[i], outnextparents[i], rel_tol=0.0001)
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert math.isclose(outindex[i], outoutindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarry = (ctypes.c_int64*len(nextcarry))(*nextcarry)
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = (ctypes.c_int64*len(nextparents))(*nextparents)
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_int32*len(index))(*index)
    indexoffset = 1
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_reduce_next_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length)
    outnextcarry = [0, 0, 1]
    for i in range(len(outnextcarry)):
        assert math.isclose(nextcarry[i], outnextcarry[i], rel_tol=0.0001)
    outnextparents = [1, 2, 3]
    for i in range(len(outnextparents)):
        assert math.isclose(nextparents[i], outnextparents[i], rel_tol=0.0001)
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert math.isclose(outindex[i], outoutindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray_reduce_next_fix_offsets_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outoffsets = (ctypes.c_int64*len(outoffsets))(*outoffsets)
    starts = [1, 0, 0, 1, 1, 1, 0]
    starts = (ctypes.c_int64*len(starts))(*starts)
    startsoffset = 1
    startslength = 3
    outindexlength = 3
    funcC = getattr(lib, 'awkward_IndexedArray_reduce_next_fix_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(outoffsets, starts, startsoffset, startslength, outindexlength)
    outoutoffsets = [0, 0, 1, 0, 3]
    for i in range(len(outoutoffsets)):
        assert math.isclose(outoffsets[i], outoutoffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_reduce_mask_ByteMaskedArray_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int8*len(toptr))(*toptr)
    parents = [1, 0, 0, 1, 1, 1, 0]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 1
    lenparents = 3
    outlength = 3
    funcC = getattr(lib, 'awkward_NumpyArray_reduce_mask_ByteMaskedArray_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 1]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarry = (ctypes.c_int64*len(nextcarry))(*nextcarry)
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = (ctypes.c_int64*len(nextparents))(*nextparents)
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    mask = [1, 1, 1, 1, 1]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 0
    parents = [1, 0, 0, 1, 1, 1, 0]
    parents = (ctypes.c_int64*len(parents))(*parents)
    parentsoffset = 1
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_reduce_next_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(nextcarry, nextparents, outindex, mask, maskoffset, parents, parentsoffset, length, validwhen)
    outnextcarry = [0, 1, 2]
    for i in range(len(outnextcarry)):
        assert math.isclose(nextcarry[i], outnextcarry[i], rel_tol=0.0001)
    outnextparents = [0, 0, 1]
    for i in range(len(outnextparents)):
        assert math.isclose(nextparents[i], outnextparents[i], rel_tol=0.0001)
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert math.isclose(outindex[i], outoutindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index8_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    length = 3
    funcC = getattr(lib, 'awkward_Index8_to_Index64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexU8_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint8*len(fromptr))(*fromptr)
    length = 3
    funcC = getattr(lib, 'awkward_IndexU8_to_Index64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index32_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_int32*len(fromptr))(*fromptr)
    length = 3
    funcC = getattr(lib, 'awkward_Index32_to_Index64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexU32_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptr = (ctypes.c_uint32*len(fromptr))(*fromptr)
    length = 3
    funcC = getattr(lib, 'awkward_IndexU32_to_Index64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexU8_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint8*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_uint8*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcC = getattr(lib, 'awkward_IndexU8_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index32_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int32*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcC = getattr(lib, 'awkward_Index32_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index64_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcC = getattr(lib, 'awkward_Index64_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexU32_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint32*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcC = getattr(lib, 'awkward_IndexU32_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index8_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int8*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_int8*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcC = getattr(lib, 'awkward_Index8_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index64_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_Index64_carry_nocheck_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index32_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int32*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_Index32_carry_nocheck_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexU8_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint8*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_uint8*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_IndexU8_carry_nocheck_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexU32_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint32*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_IndexU32_carry_nocheck_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Index8_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int8*len(toindex))(*toindex)
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    fromindex = (ctypes.c_int8*len(fromindex))(*fromindex)
    carry = [1, 0, 0, 1, 1, 1, 0]
    carry = (ctypes.c_int64*len(carry))(*carry)
    fromindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_Index8_carry_nocheck_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_slicemissing_check_same_1():
    same = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    same = (ctypes.c_bool*len(same))(*same)
    bytemask = [1, 1, 1, 1, 1]
    bytemask = (ctypes.c_int8*len(bytemask))(*bytemask)
    bytemaskoffset = 0
    missingindex = [1, 1, 1, 1, 1]
    missingindex = (ctypes.c_int64*len(missingindex))(*missingindex)
    missingindexoffset = 0
    length = 3
    funcC = getattr(lib, 'awkward_slicemissing_check_same')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(same, bytemask, bytemaskoffset, missingindex, missingindexoffset, length)
    outsame = [False]
    for i in range(len(outsame)):
        assert math.isclose(same[i], outsame[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_carry_arange32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int32*len(toptr))(*toptr)
    length = 3
    funcC = getattr(lib, 'awkward_carry_arange32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64)
    ret_pass = funcC(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_carry_arange64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    length = 3
    funcC = getattr(lib, 'awkward_carry_arange64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_carry_arangeU32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_uint32*len(toptr))(*toptr)
    length = 3
    funcC = getattr(lib, 'awkward_carry_arangeU32')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64)
    ret_pass = funcC(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities64_getitem_carry_64_1():
    newidentitiesptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    newidentitiesptr = (ctypes.c_int64*len(newidentitiesptr))(*newidentitiesptr)
    identitiesptr = [1, 0, 0, 1, 1, 1, 0]
    identitiesptr = (ctypes.c_int64*len(identitiesptr))(*identitiesptr)
    carryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    lencarry = 3
    offset = 1
    width = 3
    length = 3
    funcC = getattr(lib, 'awkward_Identities64_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length)
    outnewidentitiesptr = [0, 0, 1, 0, 0, 1, 0, 0, 1]
    for i in range(len(outnewidentitiesptr)):
        assert math.isclose(newidentitiesptr[i], outnewidentitiesptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Identities32_getitem_carry_64_1():
    newidentitiesptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    newidentitiesptr = (ctypes.c_int32*len(newidentitiesptr))(*newidentitiesptr)
    identitiesptr = [1, 0, 0, 1, 1, 1, 0]
    identitiesptr = (ctypes.c_int32*len(identitiesptr))(*identitiesptr)
    carryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    lencarry = 3
    offset = 1
    width = 3
    length = 3
    funcC = getattr(lib, 'awkward_Identities32_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length)
    outnewidentitiesptr = [0, 0, 1, 0, 0, 1, 0, 0, 1]
    for i in range(len(outnewidentitiesptr)):
        assert math.isclose(newidentitiesptr[i], outnewidentitiesptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_contiguous_init_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    skip = 3
    stride = 3
    funcC = getattr(lib, 'awkward_NumpyArray_contiguous_init_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, skip, stride)
    outtoptr = [0, 3, 6]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_contiguous_next_64_1():
    topos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    topos = (ctypes.c_int64*len(topos))(*topos)
    frompos = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    frompos = (ctypes.c_int64*len(frompos))(*frompos)
    length = 3
    skip = 3
    stride = 3
    funcC = getattr(lib, 'awkward_NumpyArray_contiguous_next_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(topos, frompos, length, skip, stride)
    outtopos = [5, 8, 11, 8, 11, 14, 7, 10, 13]
    for i in range(len(outtopos)):
        assert math.isclose(topos[i], outtopos[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_next_at_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarryptr = (ctypes.c_int64*len(nextcarryptr))(*nextcarryptr)
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    lencarry = 3
    skip = 3
    at = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_next_at_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarryptr, carryptr, lencarry, skip, at)
    outnextcarryptr = [18, 27, 24]
    for i in range(len(outnextcarryptr)):
        assert math.isclose(nextcarryptr[i], outnextcarryptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_next_range_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarryptr = (ctypes.c_int64*len(nextcarryptr))(*nextcarryptr)
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    lencarry = 3
    lenhead = 3
    skip = 3
    start = 3
    step = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_next_range_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarryptr, carryptr, lencarry, lenhead, skip, start, step)
    outnextcarryptr = [18, 21, 24, 27, 30, 33, 24, 27, 30]
    for i in range(len(outnextcarryptr)):
        assert math.isclose(nextcarryptr[i], outnextcarryptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_next_range_advanced_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarryptr = (ctypes.c_int64*len(nextcarryptr))(*nextcarryptr)
    nextadvancedptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextadvancedptr = (ctypes.c_int64*len(nextadvancedptr))(*nextadvancedptr)
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    advancedptr = [2, 1, 4, 3, 5, 2, 10, 11, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 6, 2, 3]
    advancedptr = (ctypes.c_int64*len(advancedptr))(*advancedptr)
    lencarry = 3
    lenhead = 3
    skip = 3
    start = 3
    step = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_next_range_advanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarryptr, nextadvancedptr, carryptr, advancedptr, lencarry, lenhead, skip, start, step)
    outnextcarryptr = [18, 21, 24, 27, 30, 33, 24, 27, 30]
    for i in range(len(outnextcarryptr)):
        assert math.isclose(nextcarryptr[i], outnextcarryptr[i], rel_tol=0.0001)
    outnextadvancedptr = [2, 2, 2, 1, 1, 1, 4, 4, 4]
    for i in range(len(outnextadvancedptr)):
        assert math.isclose(nextadvancedptr[i], outnextadvancedptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_next_array_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarryptr = (ctypes.c_int64*len(nextcarryptr))(*nextcarryptr)
    nextadvancedptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextadvancedptr = (ctypes.c_int64*len(nextadvancedptr))(*nextadvancedptr)
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    flatheadptr = [2, 1, 4, 3, 5, 2, 10, 11, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 6, 2, 3]
    flatheadptr = (ctypes.c_int64*len(flatheadptr))(*flatheadptr)
    lencarry = 3
    lenflathead = 3
    skip = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_next_array_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarryptr, nextadvancedptr, carryptr, flatheadptr, lencarry, lenflathead, skip)
    outnextcarryptr = [17, 16, 19, 26, 25, 28, 23, 22, 25]
    for i in range(len(outnextcarryptr)):
        assert math.isclose(nextcarryptr[i], outnextcarryptr[i], rel_tol=0.0001)
    outnextadvancedptr = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outnextadvancedptr)):
        assert math.isclose(nextadvancedptr[i], outnextadvancedptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_next_array_advanced_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextcarryptr = (ctypes.c_int64*len(nextcarryptr))(*nextcarryptr)
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    carryptr = (ctypes.c_int64*len(carryptr))(*carryptr)
    advancedptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    advancedptr = (ctypes.c_int64*len(advancedptr))(*advancedptr)
    flatheadptr = [2, 1, 4, 3, 5, 2, 10, 11, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 6, 2, 3]
    flatheadptr = (ctypes.c_int64*len(flatheadptr))(*flatheadptr)
    lencarry = 3
    skip = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_next_array_advanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(nextcarryptr, carryptr, advancedptr, flatheadptr, lencarry, skip)
    outnextcarryptr = [17, 44, 32]
    for i in range(len(outnextcarryptr)):
        assert math.isclose(nextcarryptr[i], outnextcarryptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_boolean_numtrue_1():
    numtrue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numtrue = (ctypes.c_int64*len(numtrue))(*numtrue)
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    byteoffset = 0
    length = 3
    stride = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_boolean_numtrue')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(numtrue, fromptr, byteoffset, length, stride)
    outnumtrue = [1]
    for i in range(len(outnumtrue)):
        assert math.isclose(numtrue[i], outnumtrue[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_NumpyArray_getitem_boolean_nonzero_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toptr = (ctypes.c_int64*len(toptr))(*toptr)
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromptr = (ctypes.c_int8*len(fromptr))(*fromptr)
    byteoffset = 0
    length = 3
    stride = 3
    funcC = getattr(lib, 'awkward_NumpyArray_getitem_boolean_nonzero_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toptr, fromptr, byteoffset, length, stride)
    outtoptr = [0]
    for i in range(len(outtoptr)):
        assert math.isclose(toptr[i], outtoptr[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    at = 0
    funcC = getattr(lib, 'awkward_ListArray64_getitem_next_at_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    at = 0
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_next_at_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    at = 0
    funcC = getattr(lib, 'awkward_ListArray32_getitem_next_at_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_next_range_counts_64_1():
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = (ctypes.c_int64*len(total))(*total)
    fromoffsets = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    lenstarts = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_next_range_counts_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(total, fromoffsets, lenstarts)
    outtotal = [-1]
    for i in range(len(outtotal)):
        assert math.isclose(total[i], outtotal[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_next_range_counts_64_1():
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = (ctypes.c_int64*len(total))(*total)
    fromoffsets = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    lenstarts = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_next_range_counts_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64)
    ret_pass = funcC(total, fromoffsets, lenstarts)
    outtotal = [-1]
    for i in range(len(outtotal)):
        assert math.isclose(total[i], outtotal[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_next_range_counts_64_1():
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = (ctypes.c_int64*len(total))(*total)
    fromoffsets = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    lenstarts = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_next_range_counts_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64)
    ret_pass = funcC(total, fromoffsets, lenstarts)
    outtotal = [-1]
    for i in range(len(outtotal)):
        assert math.isclose(total[i], outtotal[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromadvanced = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    fromoffsets = [1, 2, 3, 4, 5, 6]
    fromoffsets = (ctypes.c_uint32*len(fromoffsets))(*fromoffsets)
    lenstarts = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_next_range_spreadadvanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64)
    ret_pass = funcC(toadvanced, fromadvanced, fromoffsets, lenstarts)
    outtoadvanced = [0, 2, 0, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromadvanced = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    fromoffsets = [1, 2, 3, 4, 5, 6]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    lenstarts = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_next_range_spreadadvanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(toadvanced, fromadvanced, fromoffsets, lenstarts)
    outtoadvanced = [0, 2, 0, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromadvanced = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    fromoffsets = [1, 2, 3, 4, 5, 6]
    fromoffsets = (ctypes.c_int32*len(fromoffsets))(*fromoffsets)
    lenstarts = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_next_range_spreadadvanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64)
    ret_pass = funcC(toadvanced, fromadvanced, fromoffsets, lenstarts)
    outtoadvanced = [0, 2, 0, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_next_array_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 2, 2, 0, 0, 0, 2, 2, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_next_array_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 2, 2, 0, 0, 0, 2, 2, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_next_array_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 2, 2, 0, 0, 0, 2, 2, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    fromadvanced = [1, 2, 3, 4, 5, 6]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_next_array_advanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    fromadvanced = [1, 2, 3, 4, 5, 6]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_next_array_advanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    fromadvanced = [1, 2, 3, 4, 5, 6]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_next_array_advanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_carry_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int64*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int64*len(tostops))(*tostops)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lencarry = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry)
    outtostarts = [2.0, 2.0, 2.0]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3.0, 3.0, 3.0]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_carry_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_int32*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_int32*len(tostops))(*tostops)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lencarry = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry)
    outtostarts = [2.0, 2.0, 2.0]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3.0, 3.0, 3.0]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_carry_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostarts = (ctypes.c_uint32*len(tostarts))(*tostarts)
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = (ctypes.c_uint32*len(tostops))(*tostops)
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstops = [3, 1, 3, 2, 3]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lencarry = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry)
    outtostarts = [2.0, 2.0, 2.0]
    for i in range(len(outtostarts)):
        assert math.isclose(tostarts[i], outtostarts[i], rel_tol=0.0001)
    outtostops = [3.0, 3.0, 3.0]
    for i in range(len(outtostops)):
        assert math.isclose(tostops[i], outtostops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    at = 0
    length = 3
    size = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_next_at_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, at, length, size)
    outtocarry = [0, 3, 6]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_next_range_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    regular_start = 3
    step = 3
    length = 3
    size = 3
    nextsize = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_next_range_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, regular_start, step, length, size, nextsize)
    outtocarry = [3, 6, 9, 6, 9, 12, 9, 12, 15]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromadvanced = [0, 3, 6, 9, 12, 15]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    length = 3
    nextsize = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_next_range_spreadadvanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toadvanced, fromadvanced, length, nextsize)
    outtoadvanced = [0, 0, 0, 3, 3, 3, 6, 6, 6]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_next_array_regularize_64_1():
    toarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toarray = (ctypes.c_int64*len(toarray))(*toarray)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    lenarray = 3
    size = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_next_array_regularize_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toarray, fromarray, lenarray, size)
    outtoarray = [0, 0, 0]
    for i in range(len(outtoarray)):
        assert math.isclose(toarray[i], outtoarray[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    length = 3
    lenarray = 3
    size = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_next_array_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromarray, length, lenarray, size)
    outtocarry = [0, 0, 0, 3, 3, 3, 6, 6, 6]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = (ctypes.c_int64*len(toadvanced))(*toadvanced)
    fromadvanced = [0, 3, 6, 9, 12, 15]
    fromadvanced = (ctypes.c_int64*len(fromadvanced))(*fromadvanced)
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = (ctypes.c_int64*len(fromarray))(*fromarray)
    length = 3
    lenarray = 3
    size = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_next_array_advanced_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toadvanced, fromadvanced, fromarray, length, lenarray, size)
    outtocarry = [0, 3, 6]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert math.isclose(toadvanced[i], outtoadvanced[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_carry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromcarry = [0, 3, 6, 9, 12, 15]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    lencarry = 3
    size = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromcarry, lencarry, size)
    outtocarry = [0, 1, 2, 9, 10, 11, 18, 19, 20]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numnull = (ctypes.c_int64*len(numnull))(*numnull)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_numnull')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(numnull, fromindex, indexoffset, lenindex)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert math.isclose(numnull[i], outnumnull[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numnull = (ctypes.c_int64*len(numnull))(*numnull)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_numnull')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(numnull, fromindex, indexoffset, lenindex)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert math.isclose(numnull[i], outnumnull[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numnull = (ctypes.c_int64*len(numnull))(*numnull)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_numnull')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(numnull, fromindex, indexoffset, lenindex)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert math.isclose(numnull[i], outnumnull[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint32*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_getitem_nextcarry_outindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int32*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_getitem_nextcarry_outindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_getitem_nextcarry_outindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_getitem_adjust_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tononzero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tononzero = (ctypes.c_int64*len(tononzero))(*tononzero)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    length = 3
    nonzero = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    nonzero = (ctypes.c_int64*len(nonzero))(*nonzero)
    nonzerooffset = 0
    nonzerolength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_getitem_adjust_offsets_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, tononzero, fromoffsets, offsetsoffset, length, nonzero, nonzerooffset, nonzerolength)
    outtooffsets = [1, 1, 1, 1]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtononzero = []
    for i in range(len(outtononzero)):
        assert math.isclose(tononzero[i], outtononzero[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListOffsetArray_getitem_adjust_offsets_index_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tononzero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tononzero = (ctypes.c_int64*len(tononzero))(*tononzero)
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    fromoffsets = (ctypes.c_int64*len(fromoffsets))(*fromoffsets)
    offsetsoffset = 1
    length = 3
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_int64*len(index))(*index)
    indexoffset = 1
    indexlength = 3
    nonzero = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    nonzero = (ctypes.c_int64*len(nonzero))(*nonzero)
    nonzerooffset = 0
    nonzerolength = 3
    originalmask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    originalmask = (ctypes.c_int8*len(originalmask))(*originalmask)
    maskoffset = 1
    masklength = 3
    funcC = getattr(lib, 'awkward_ListOffsetArray_getitem_adjust_offsets_index_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, tononzero, fromoffsets, offsetsoffset, length, index, indexoffset, indexlength, nonzero, nonzerooffset, nonzerolength, originalmask, maskoffset, masklength)
    outtooffsets = [1, 1, 1, 1]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtononzero = []
    for i in range(len(outtononzero)):
        assert math.isclose(tononzero[i], outtononzero[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray_getitem_adjust_outindex_64_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    tononzero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tononzero = (ctypes.c_int64*len(tononzero))(*tononzero)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromindexoffset = 1
    fromindexlength = 3
    nonzero = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    nonzero = (ctypes.c_int64*len(nonzero))(*nonzero)
    nonzerooffset = 0
    nonzerolength = 3
    funcC = getattr(lib, 'awkward_IndexedArray_getitem_adjust_outindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tomask, toindex, tononzero, fromindex, fromindexoffset, fromindexlength, nonzero, nonzerooffset, nonzerolength)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    outtoindex = []
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outtononzero = []
    for i in range(len(outtononzero)):
        assert math.isclose(tononzero[i], outtononzero[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_getitem_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_getitem_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_getitem_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArrayU32_getitem_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint32*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    indexoffset = 1
    lenindex = 3
    lencarry = 3
    funcC = getattr(lib, 'awkward_IndexedArrayU32_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry)
    outtoindex = [0.0, 0.0, 0.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray64_getitem_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    indexoffset = 1
    lenindex = 3
    lencarry = 3
    funcC = getattr(lib, 'awkward_IndexedArray64_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry)
    outtoindex = [0.0, 0.0, 0.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_IndexedArray32_getitem_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int32*len(toindex))(*toindex)
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    indexoffset = 1
    lenindex = 3
    lencarry = 3
    funcC = getattr(lib, 'awkward_IndexedArray32_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry)
    outtoindex = [0.0, 0.0, 0.0]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_regular_index_getsize_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = (ctypes.c_int64*len(size))(*size)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray8_regular_index_getsize')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(size, fromtags, tagsoffset, length)
    outsize = [1]
    for i in range(len(outsize)):
        assert math.isclose(size[i], outsize[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_regular_index_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int32*len(toindex))(*toindex)
    current = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    current = (ctypes.c_int32*len(current))(*current)
    size = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray8_32_regular_index')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, current, size, fromtags, tagsoffset, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outcurrent = [3, 0, 0]
    for i in range(len(outcurrent)):
        assert math.isclose(current[i], outcurrent[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_U32_regular_index_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_uint32*len(toindex))(*toindex)
    current = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    current = (ctypes.c_uint32*len(current))(*current)
    size = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray8_U32_regular_index')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, current, size, fromtags, tagsoffset, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outcurrent = [3, 0, 0]
    for i in range(len(outcurrent)):
        assert math.isclose(current[i], outcurrent[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_64_regular_index_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    current = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    current = (ctypes.c_int64*len(current))(*current)
    size = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    length = 3
    funcC = getattr(lib, 'awkward_UnionArray8_64_regular_index')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(toindex, current, size, fromtags, tagsoffset, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    outcurrent = [3, 0, 0]
    for i in range(len(outcurrent)):
        assert math.isclose(current[i], outcurrent[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_U32_project_64_1():
    lenout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lenout = (ctypes.c_int64*len(lenout))(*lenout)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_uint32*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    which = 1
    funcC = getattr(lib, 'awkward_UnionArray8_U32_project_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which)
    outlenout = [0]
    for i in range(len(outlenout)):
        assert math.isclose(lenout[i], outlenout[i], rel_tol=0.0001)
    outtocarry = []
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_64_project_64_1():
    lenout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lenout = (ctypes.c_int64*len(lenout))(*lenout)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int64*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    which = 1
    funcC = getattr(lib, 'awkward_UnionArray8_64_project_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which)
    outlenout = [0]
    for i in range(len(outlenout)):
        assert math.isclose(lenout[i], outlenout[i], rel_tol=0.0001)
    outtocarry = []
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_UnionArray8_32_project_64_1():
    lenout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lenout = (ctypes.c_int64*len(lenout))(*lenout)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = (ctypes.c_int8*len(fromtags))(*fromtags)
    tagsoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindex = (ctypes.c_int32*len(fromindex))(*fromindex)
    indexoffset = 1
    length = 3
    which = 1
    funcC = getattr(lib, 'awkward_UnionArray8_32_project_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which)
    outlenout = [0]
    for i in range(len(outlenout)):
        assert math.isclose(lenout[i], outlenout[i], rel_tol=0.0001)
    outtocarry = []
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_missing_repeat_64_1():
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_int64*len(index))(*index)
    indexoffset = 1
    indexlength = 3
    repetitions = 3
    regularsize = 3
    funcC = getattr(lib, 'awkward_missing_repeat_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(outindex, index, indexoffset, indexlength, repetitions, regularsize)
    outoutindex = [0, 0, 1, 3, 3, 4, 6, 6, 7]
    for i in range(len(outoutindex)):
        assert math.isclose(outindex[i], outoutindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_RegularArray_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistarts = (ctypes.c_int64*len(multistarts))(*multistarts)
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = (ctypes.c_int64*len(multistops))(*multistops)
    singleoffsets = [0, 3, 6, 9, 12, 15]
    singleoffsets = (ctypes.c_int64*len(singleoffsets))(*singleoffsets)
    regularsize = 3
    regularlength = 3
    funcC = getattr(lib, 'awkward_RegularArray_getitem_jagged_expand_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(multistarts, multistops, singleoffsets, regularsize, regularlength)
    outmultistarts = [0, 3, 6, 0, 3, 6, 0, 3, 6]
    for i in range(len(outmultistarts)):
        assert math.isclose(multistarts[i], outmultistarts[i], rel_tol=0.0001)
    outmultistops = [3, 6, 9, 3, 6, 9, 3, 6, 9]
    for i in range(len(outmultistops)):
        assert math.isclose(multistops[i], outmultistops[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistarts = (ctypes.c_int64*len(multistarts))(*multistarts)
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = (ctypes.c_int64*len(multistops))(*multistops)
    singleoffsets = [1, 2, 3, 4, 5, 6]
    singleoffsets = (ctypes.c_int64*len(singleoffsets))(*singleoffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    jaggedsize = 2
    length = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_jagged_expand_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, jaggedsize, length)
    outmultistarts = [1, 2, 1, 2, 1, 2]
    for i in range(len(outmultistarts)):
        assert math.isclose(multistarts[i], outmultistarts[i], rel_tol=0.0001)
    outmultistops = [2, 3, 2, 3, 2, 3]
    for i in range(len(outmultistops)):
        assert math.isclose(multistops[i], outmultistops[i], rel_tol=0.0001)
    outtocarry = [1, 2, 2, 3, 3, 4]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistarts = (ctypes.c_int64*len(multistarts))(*multistarts)
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = (ctypes.c_int64*len(multistops))(*multistops)
    singleoffsets = [1, 2, 3, 4, 5, 6]
    singleoffsets = (ctypes.c_int64*len(singleoffsets))(*singleoffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    jaggedsize = 2
    length = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_jagged_expand_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, jaggedsize, length)
    outmultistarts = [1, 2, 1, 2, 1, 2]
    for i in range(len(outmultistarts)):
        assert math.isclose(multistarts[i], outmultistarts[i], rel_tol=0.0001)
    outmultistops = [2, 3, 2, 3, 2, 3]
    for i in range(len(outmultistops)):
        assert math.isclose(multistops[i], outmultistops[i], rel_tol=0.0001)
    outtocarry = [1, 2, 2, 3, 3, 4]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistarts = (ctypes.c_int64*len(multistarts))(*multistarts)
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = (ctypes.c_int64*len(multistops))(*multistops)
    singleoffsets = [1, 2, 3, 4, 5, 6]
    singleoffsets = (ctypes.c_int64*len(singleoffsets))(*singleoffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    jaggedsize = 2
    length = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_jagged_expand_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, jaggedsize, length)
    outmultistarts = [1, 2, 1, 2, 1, 2]
    for i in range(len(outmultistarts)):
        assert math.isclose(multistarts[i], outmultistarts[i], rel_tol=0.0001)
    outmultistops = [2, 3, 2, 3, 2, 3]
    for i in range(len(outmultistops)):
        assert math.isclose(multistops[i], outmultistops[i], rel_tol=0.0001)
    outtocarry = [1, 2, 2, 3, 3, 4]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray_getitem_jagged_carrylen_64_1():
    carrylen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carrylen = (ctypes.c_int64*len(carrylen))(*carrylen)
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    funcC = getattr(lib, 'awkward_ListArray_getitem_jagged_carrylen_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(carrylen, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen)
    outcarrylen = [3]
    for i in range(len(outcarrylen)):
        assert math.isclose(carrylen[i], outcarrylen[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_jagged_apply_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sliceindex = (ctypes.c_int64*len(sliceindex))(*sliceindex)
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    contentlen = 10
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_jagged_apply_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
    outtooffsets = [0.0, 1.0, 2.0, 3.0]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtocarry = [1, 2, 3]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_jagged_apply_64_2():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    slicestarts = [3, 0, 3, 2, 3]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sliceindex = (ctypes.c_int64*len(sliceindex))(*sliceindex)
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [2, 3, 4, 5, 6, 7]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    contentlen = 4
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_jagged_apply_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.c_int64)
    assert funcC(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen).str.contents

def test_awkward_ListArray64_getitem_jagged_apply_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sliceindex = (ctypes.c_int64*len(sliceindex))(*sliceindex)
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    contentlen = 10
    funcC = getattr(lib, 'awkward_ListArray64_getitem_jagged_apply_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
    outtooffsets = [0.0, 1.0, 2.0, 3.0]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtocarry = [1, 2, 3]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_jagged_apply_64_2():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    slicestarts = [3, 0, 3, 2, 3]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sliceindex = (ctypes.c_int64*len(sliceindex))(*sliceindex)
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [2, 3, 4, 5, 6, 7]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    contentlen = 4
    funcC = getattr(lib, 'awkward_ListArray64_getitem_jagged_apply_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    assert funcC(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen).str.contents

def test_awkward_ListArray32_getitem_jagged_apply_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sliceindex = (ctypes.c_int64*len(sliceindex))(*sliceindex)
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    contentlen = 10
    funcC = getattr(lib, 'awkward_ListArray32_getitem_jagged_apply_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
    outtooffsets = [0.0, 1.0, 2.0, 3.0]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    outtocarry = [1, 2, 3]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_jagged_apply_64_2():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    slicestarts = [3, 0, 3, 2, 3]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sliceindex = (ctypes.c_int64*len(sliceindex))(*sliceindex)
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 0
    fromstops = [2, 3, 4, 5, 6, 7]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromstopsoffset = 0
    contentlen = 4
    funcC = getattr(lib, 'awkward_ListArray32_getitem_jagged_apply_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int64)
    assert funcC(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen).str.contents

def test_awkward_ListArray_getitem_jagged_numvalid_64_1():
    numvalid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numvalid = (ctypes.c_int64*len(numvalid))(*numvalid)
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    length = 3
    missing = [1, 2, 3, 4, 5, 6]
    missing = (ctypes.c_int64*len(missing))(*missing)
    missingoffset = 0
    missinglength = 3
    funcC = getattr(lib, 'awkward_ListArray_getitem_jagged_numvalid_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    ret_pass = funcC(numvalid, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, length, missing, missingoffset, missinglength)
    outnumvalid = [3]
    for i in range(len(outnumvalid)):
        assert math.isclose(numvalid[i], outnumvalid[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray_getitem_jagged_numvalid_64_2():
    numvalid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numvalid = (ctypes.c_int64*len(numvalid))(*numvalid)
    slicestarts = [3, 0, 3, 2, 3]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    length = 3
    missing = [0, 4, 1, 2, 3, 3]
    missing = (ctypes.c_int64*len(missing))(*missing)
    missingoffset = 0
    missinglength = 3
    funcC = getattr(lib, 'awkward_ListArray_getitem_jagged_numvalid_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64)
    assert funcC(numvalid, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, length, missing, missingoffset, missinglength).str.contents

def test_awkward_ListArray_getitem_jagged_shrink_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    tosmalloffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tosmalloffsets = (ctypes.c_int64*len(tosmalloffsets))(*tosmalloffsets)
    tolargeoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tolargeoffsets = (ctypes.c_int64*len(tolargeoffsets))(*tolargeoffsets)
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    length = 3
    missing = [1, 2, 3, 4, 5, 6]
    missing = (ctypes.c_int64*len(missing))(*missing)
    missingoffset = 0
    funcC = getattr(lib, 'awkward_ListArray_getitem_jagged_shrink_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(tocarry, tosmalloffsets, tolargeoffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, length, missing, missingoffset)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outtosmalloffsets = [2, 3, 4, 5]
    for i in range(len(outtosmalloffsets)):
        assert math.isclose(tosmalloffsets[i], outtosmalloffsets[i], rel_tol=0.0001)
    outtolargeoffsets = [2, 3, 4, 5]
    for i in range(len(outtolargeoffsets)):
        assert math.isclose(tolargeoffsets[i], outtolargeoffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArrayU32_getitem_jagged_descend_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    slicestarts = [1, 2, 3, 4, 5, 6]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 4, 5, 6, 7, 8]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    fromstarts = [2, 3, 4, 5, 6, 7]
    fromstarts = (ctypes.c_uint32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 3
    fromstops = [4, 5, 6, 7, 8, 9]
    fromstops = (ctypes.c_uint32*len(fromstops))(*fromstops)
    fromstopsoffset = 3
    funcC = getattr(lib, 'awkward_ListArrayU32_getitem_jagged_descend_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64)
    ret_pass = funcC(tooffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset)
    outtooffsets = [1, 3.0, 5.0, 7.0]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray64_getitem_jagged_descend_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    slicestarts = [1, 2, 3, 4, 5, 6]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 4, 5, 6, 7, 8]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    fromstarts = [2, 3, 4, 5, 6, 7]
    fromstarts = (ctypes.c_int64*len(fromstarts))(*fromstarts)
    fromstartsoffset = 3
    fromstops = [4, 5, 6, 7, 8, 9]
    fromstops = (ctypes.c_int64*len(fromstops))(*fromstops)
    fromstopsoffset = 3
    funcC = getattr(lib, 'awkward_ListArray64_getitem_jagged_descend_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(tooffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset)
    outtooffsets = [1, 3.0, 5.0, 7.0]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ListArray32_getitem_jagged_descend_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = (ctypes.c_int64*len(tooffsets))(*tooffsets)
    slicestarts = [1, 2, 3, 4, 5, 6]
    slicestarts = (ctypes.c_int64*len(slicestarts))(*slicestarts)
    slicestartsoffset = 0
    slicestops = [3, 4, 5, 6, 7, 8]
    slicestops = (ctypes.c_int64*len(slicestops))(*slicestops)
    slicestopsoffset = 0
    sliceouterlen = 3
    fromstarts = [2, 3, 4, 5, 6, 7]
    fromstarts = (ctypes.c_int32*len(fromstarts))(*fromstarts)
    fromstartsoffset = 3
    fromstops = [4, 5, 6, 7, 8, 9]
    fromstops = (ctypes.c_int32*len(fromstops))(*fromstops)
    fromstopsoffset = 3
    funcC = getattr(lib, 'awkward_ListArray32_getitem_jagged_descend_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64)
    ret_pass = funcC(tooffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset)
    outtooffsets = [1, 3.0, 5.0, 7.0]
    for i in range(len(outtooffsets)):
        assert math.isclose(tooffsets[i], outtooffsets[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_getitem_carry_64_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomask = (ctypes.c_int8*len(tomask))(*tomask)
    frommask = [1, 1, 1, 1, 1]
    frommask = (ctypes.c_int8*len(frommask))(*frommask)
    frommaskoffset = 0
    lenmask = 3
    fromcarry = [1, 0, 0, 1, 1, 1, 0]
    fromcarry = (ctypes.c_int64*len(fromcarry))(*fromcarry)
    lencarry = 3
    funcC = getattr(lib, 'awkward_ByteMaskedArray_getitem_carry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(tomask, frommask, frommaskoffset, lenmask, fromcarry, lencarry)
    outtomask = [1, 1, 1]
    for i in range(len(outtomask)):
        assert math.isclose(tomask[i], outtomask[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    numnull = (ctypes.c_int64*len(numnull))(*numnull)
    mask = [1, 1, 1, 1, 1]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 0
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_numnull')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(numnull, mask, maskoffset, length, validwhen)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert math.isclose(numnull[i], outnumnull[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    mask = [1, 1, 1, 1, 1]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 0
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_getitem_nextcarry_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(tocarry, mask, maskoffset, length, validwhen)
    outtocarry = [0, 1, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = (ctypes.c_int64*len(tocarry))(*tocarry)
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = (ctypes.c_int64*len(outindex))(*outindex)
    mask = [1, 1, 1, 1, 1]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 0
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_getitem_nextcarry_outindex_64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(tocarry, outindex, mask, maskoffset, length, validwhen)
    outtocarry = [0, 1, 2]
    for i in range(len(outtocarry)):
        assert math.isclose(tocarry[i], outtocarry[i], rel_tol=0.0001)
    outoutindex = [0.0, 1.0, 2.0]
    for i in range(len(outoutindex)):
        assert math.isclose(outindex[i], outoutindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_ByteMaskedArray_toIndexedOptionArray64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = (ctypes.c_int64*len(toindex))(*toindex)
    mask = [1, 1, 1, 1, 1]
    mask = (ctypes.c_int8*len(mask))(*mask)
    maskoffset = 0
    length = 3
    validwhen = True
    funcC = getattr(lib, 'awkward_ByteMaskedArray_toIndexedOptionArray64')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int64, ctypes.c_bool)
    ret_pass = funcC(toindex, mask, maskoffset, length, validwhen)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert math.isclose(toindex[i], outtoindex[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_Content_getitem_next_missing_jagged_getmaskstartstop_1():
    index_in = [1, 0, 0, 1, 1, 1, 0]
    index_in = (ctypes.c_int64*len(index_in))(*index_in)
    index_in_offset = 1
    offsets_in = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsets_in = (ctypes.c_int64*len(offsets_in))(*offsets_in)
    offsets_in_offset = 0
    mask_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask_out = (ctypes.c_int64*len(mask_out))(*mask_out)
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    starts_out = (ctypes.c_int64*len(starts_out))(*starts_out)
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = (ctypes.c_int64*len(stops_out))(*stops_out)
    length = 3
    funcC = getattr(lib, 'awkward_Content_getitem_next_missing_jagged_getmaskstartstop')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(index_in, index_in_offset, offsets_in, offsets_in_offset, mask_out, starts_out, stops_out, length)
    outmask_out = [0, 1, 2]
    for i in range(len(outmask_out)):
        assert math.isclose(mask_out[i], outmask_out[i], rel_tol=0.0001)
    outstarts_out = [14, 1, 27]
    for i in range(len(outstarts_out)):
        assert math.isclose(starts_out[i], outstarts_out[i], rel_tol=0.0001)
    outstops_out = [1, 27, 25]
    for i in range(len(outstops_out)):
        assert math.isclose(stops_out[i], outstops_out[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_MaskedArray32_getitem_next_jagged_project_1():
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_int32*len(index))(*index)
    index_offset = 1
    starts_in = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts_in = (ctypes.c_int64*len(starts_in))(*starts_in)
    starts_offset = 0
    stops_in = [3, 1, 3, 2, 3]
    stops_in = (ctypes.c_int64*len(stops_in))(*stops_in)
    stops_offset = 0
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    starts_out = (ctypes.c_int64*len(starts_out))(*starts_out)
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = (ctypes.c_int64*len(stops_out))(*stops_out)
    length = 3
    funcC = getattr(lib, 'awkward_MaskedArray32_getitem_next_jagged_project')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(index, index_offset, starts_in, starts_offset, stops_in, stops_offset, starts_out, stops_out, length)
    outstarts_out = [2, 0, 2]
    for i in range(len(outstarts_out)):
        assert math.isclose(starts_out[i], outstarts_out[i], rel_tol=0.0001)
    outstops_out = [3, 1, 3]
    for i in range(len(outstops_out)):
        assert math.isclose(stops_out[i], outstops_out[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_MaskedArray64_getitem_next_jagged_project_1():
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_int64*len(index))(*index)
    index_offset = 1
    starts_in = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts_in = (ctypes.c_int64*len(starts_in))(*starts_in)
    starts_offset = 0
    stops_in = [3, 1, 3, 2, 3]
    stops_in = (ctypes.c_int64*len(stops_in))(*stops_in)
    stops_offset = 0
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    starts_out = (ctypes.c_int64*len(starts_out))(*starts_out)
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = (ctypes.c_int64*len(stops_out))(*stops_out)
    length = 3
    funcC = getattr(lib, 'awkward_MaskedArray64_getitem_next_jagged_project')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(index, index_offset, starts_in, starts_offset, stops_in, stops_offset, starts_out, stops_out, length)
    outstarts_out = [2, 0, 2]
    for i in range(len(outstarts_out)):
        assert math.isclose(starts_out[i], outstarts_out[i], rel_tol=0.0001)
    outstops_out = [3, 1, 3]
    for i in range(len(outstops_out)):
        assert math.isclose(stops_out[i], outstops_out[i], rel_tol=0.0001)
    assert not ret_pass.str

def test_awkward_MaskedArrayU32_getitem_next_jagged_project_1():
    index = [1, 0, 0, 1, 1, 1, 0]
    index = (ctypes.c_uint32*len(index))(*index)
    index_offset = 1
    starts_in = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts_in = (ctypes.c_int64*len(starts_in))(*starts_in)
    starts_offset = 0
    stops_in = [3, 1, 3, 2, 3]
    stops_in = (ctypes.c_int64*len(stops_in))(*stops_in)
    stops_offset = 0
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    starts_out = (ctypes.c_int64*len(starts_out))(*starts_out)
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = (ctypes.c_int64*len(stops_out))(*stops_out)
    length = 3
    funcC = getattr(lib, 'awkward_MaskedArrayU32_getitem_next_jagged_project')
    funcC.restype = Error
    funcC.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
    ret_pass = funcC(index, index_offset, starts_in, starts_offset, stops_in, stops_offset, starts_out, stops_out, length)
    outstarts_out = [2, 0, 2]
    for i in range(len(outstarts_out)):
        assert math.isclose(starts_out[i], outstarts_out[i], rel_tol=0.0001)
    outstops_out = [3, 1, 3]
    for i in range(len(outstops_out)):
        assert math.isclose(stops_out[i], outstops_out[i], rel_tol=0.0001)
    assert not ret_pass.str

