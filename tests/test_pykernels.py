import tests.kernels

def test_awkward_new_Identities64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_new_Identities64')
    funcPy(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_new_Identities32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_new_Identities32')
    funcPy(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_to_Identities64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    length = 3
    width = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_to_Identities64')
    funcPy(toptr, fromptr, length, width)
    outtoptr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_ListOffsetArray64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_ListOffsetArray64')
    funcPy(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_ListOffsetArray32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_ListOffsetArray32')
    funcPy(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_ListOffsetArray64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_ListOffsetArray64')
    funcPy(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_ListOffsetArrayU32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_ListOffsetArrayU32')
    funcPy(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_ListOffsetArray32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_ListOffsetArray32')
    funcPy(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_ListOffsetArrayU32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffsets = [0, 2, 1, 1, 1, 1, 2, 0, 1, 2]
    fromptroffset = 0
    offsetsoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_ListOffsetArrayU32')
    funcPy(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth)
    outtoptr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_ListArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_ListArray32')
    funcPy(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_ListArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_ListArray64')
    funcPy(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_ListArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_ListArrayU32')
    funcPy(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_ListArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_ListArrayU32')
    funcPy(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_ListArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_ListArray64')
    funcPy(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_ListArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromptroffset = 0
    startsoffset = 0
    stopsoffset = 0
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_ListArray32')
    funcPy(uniquecontents, toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [4, 5, 6, 0.0, -1, -1, -1, -1, 1, 2, 3, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_RegularArray_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptroffset = 0
    size = 3
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_RegularArray')
    funcPy(toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth)
    outtoptr = [1, 2, 3, 0.0, 1, 2, 3, 1.0, 1, 2, 3, 2.0, 4, 5, 6, 0.0, 4, 5, 6, 1.0, 4, 5, 6, 2.0, 7, 8, 9, 0.0, 7, 8, 9, 1.0, 7, 8, 9, 2.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_RegularArray_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromptroffset = 0
    size = 3
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_RegularArray')
    funcPy(toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth)
    outtoptr = [1, 2, 3, 0.0, 1, 2, 3, 1.0, 1, 2, 3, 2.0, 4, 5, 6, 0.0, 4, 5, 6, 1.0, 4, 5, 6, 2.0, 7, 8, 9, 0.0, 7, 8, 9, 1.0, 7, 8, 9, 2.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_IndexedArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_IndexedArray32')
    funcPy(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_IndexedArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_IndexedArrayU32')
    funcPy(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_IndexedArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_IndexedArray64')
    funcPy(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_IndexedArrayU32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_IndexedArrayU32')
    funcPy(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_IndexedArray32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_IndexedArray32')
    funcPy(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_IndexedArray64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_IndexedArray64')
    funcPy(uniquecontents, toptr, fromptr, fromindex, fromptroffset, indexoffset, tolength, fromlength, fromwidth)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_UnionArray8_32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_UnionArray8_32')
    funcPy(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_UnionArray8_32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_UnionArray8_32')
    funcPy(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_UnionArray8_64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_UnionArray8_64')
    funcPy(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_UnionArray8_64_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_UnionArray8_64')
    funcPy(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_from_UnionArray8_U32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcPy = getattr(tests.kernels, 'awkward_Identities64_from_UnionArray8_U32')
    funcPy(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_from_UnionArray8_U32_1():
    uniquecontents = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 0
    tagsoffset = 1
    indexoffset = 1
    tolength = 3
    fromlength = 3
    fromwidth = 3
    which = 0
    funcPy = getattr(tests.kernels, 'awkward_Identities32_from_UnionArray8_U32')
    funcPy(uniquecontents, toptr, fromptr, fromtags, fromindex, fromptroffset, tagsoffset, indexoffset, tolength, fromlength, fromwidth, which)
    outuniquecontents = [False]
    for i in range(len(outuniquecontents)):
        assert uniquecontents[i] == outuniquecontents[i]
    outtoptr = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_extend_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffset = 0
    fromlength = 3
    tolength = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_extend')
    funcPy(toptr, fromptr, fromoffset, fromlength, tolength)
    outtoptr = [1, 2, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities32_extend_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fromoffset = 0
    fromlength = 3
    tolength = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_extend')
    funcPy(toptr, fromptr, fromoffset, fromlength, tolength)
    outtoptr = [1, 2, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_ListArrayU32_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_num_64')
    funcPy(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length)
    outtonum = [1.0, 1.0, 1.0]
    for i in range(len(outtonum)):
        assert tonum[i] == outtonum[i]

def test_awkward_ListArray32_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_num_64')
    funcPy(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length)
    outtonum = [1.0, 1.0, 1.0]
    for i in range(len(outtonum)):
        assert tonum[i] == outtonum[i]

def test_awkward_ListArray64_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_num_64')
    funcPy(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length)
    outtonum = [1.0, 1.0, 1.0]
    for i in range(len(outtonum)):
        assert tonum[i] == outtonum[i]

def test_awkward_RegularArray_num_64_1():
    tonum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_num_64')
    funcPy(tonum, size, length)
    outtonum = [3, 3, 3]
    for i in range(len(outtonum)):
        assert tonum[i] == outtonum[i]

def test_awkward_ListOffsetArrayU32_flatten_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outeroffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    outeroffsetsoffset = 1
    outeroffsetslen = 3
    inneroffsets = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    inneroffsetsoffset = 0
    inneroffsetslen = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArrayU32_flatten_offsets_64')
    funcPy(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen)
    outtooffsets = [50, 51, 55]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListOffsetArray64_flatten_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outeroffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    outeroffsetsoffset = 1
    outeroffsetslen = 3
    inneroffsets = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    inneroffsetsoffset = 0
    inneroffsetslen = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray64_flatten_offsets_64')
    funcPy(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen)
    outtooffsets = [50, 51, 55]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListOffsetArray32_flatten_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outeroffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    outeroffsetsoffset = 1
    outeroffsetslen = 3
    inneroffsets = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    inneroffsetsoffset = 0
    inneroffsetslen = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray32_flatten_offsets_64')
    funcPy(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen)
    outtooffsets = [50, 51, 55]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_IndexedArray32_flatten_none2empty_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [1, 0, 0, 1, 1, 1, 0]
    outindexoffset = 1
    outindexlength = 3
    offsets = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_flatten_none2empty_64')
    funcPy(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength)
    outoutoffsets = [14, 1, -12, 14]
    for i in range(len(outoutoffsets)):
        assert outoffsets[i] == outoutoffsets[i]

def test_awkward_IndexedArrayU32_flatten_none2empty_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [1, 0, 0, 1, 1, 1, 0]
    outindexoffset = 1
    outindexlength = 3
    offsets = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_flatten_none2empty_64')
    funcPy(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength)
    outoutoffsets = [14, 1, -12, 14]
    for i in range(len(outoutoffsets)):
        assert outoffsets[i] == outoutoffsets[i]

def test_awkward_IndexedArray64_flatten_none2empty_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [1, 0, 0, 1, 1, 1, 0]
    outindexoffset = 1
    outindexlength = 3
    offsets = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_flatten_none2empty_64')
    funcPy(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength)
    outoutoffsets = [14, 1, -12, 14]
    for i in range(len(outoutoffsets)):
        assert outoffsets[i] == outoutoffsets[i]

def test_awkward_UnionArray32_flatten_length_64_1():
    total_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray32_flatten_length_64')
    funcPy(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotal_length = [1]
    for i in range(len(outtotal_length)):
        assert total_length[i] == outtotal_length[i]

def test_awkward_UnionArrayU32_flatten_length_64_1():
    total_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    funcPy = getattr(tests.kernels, 'awkward_UnionArrayU32_flatten_length_64')
    funcPy(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotal_length = [1]
    for i in range(len(outtotal_length)):
        assert total_length[i] == outtotal_length[i]

def test_awkward_UnionArray64_flatten_length_64_1():
    total_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray64_flatten_length_64')
    funcPy(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotal_length = [1]
    for i in range(len(outtotal_length)):
        assert total_length[i] == outtotal_length[i]

def test_awkward_UnionArray64_flatten_combine_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray64_flatten_combine_64')
    funcPy(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotags = [0, 0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [1, 3]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtooffsets = [0, 1, 0, 1]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_UnionArrayU32_flatten_combine_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    funcPy = getattr(tests.kernels, 'awkward_UnionArrayU32_flatten_combine_64')
    funcPy(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotags = [0, 0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [1, 3]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtooffsets = [0, 1, 0, 1]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_UnionArray32_flatten_combine_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    offsetsraws = [[1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    offsetsoffsets = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray32_flatten_combine_64')
    funcPy(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets)
    outtotags = [0, 0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [1, 3]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtooffsets = [0, 1, 0, 1]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_IndexedArray32_flatten_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_flatten_nextcarry_64')
    funcPy(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArrayU32_flatten_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_flatten_nextcarry_64')
    funcPy(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArray64_flatten_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_flatten_nextcarry_64')
    funcPy(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArrayU32_overlay_mask8_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maskoffset = 0
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_overlay_mask8_to64')
    funcPy(toindex, mask, maskoffset, fromindex, indexoffset, length)
    outtoindex = [0, 0, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_overlay_mask8_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maskoffset = 0
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_overlay_mask8_to64')
    funcPy(toindex, mask, maskoffset, fromindex, indexoffset, length)
    outtoindex = [0, 0, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_overlay_mask8_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maskoffset = 0
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_overlay_mask8_to64')
    funcPy(toindex, mask, maskoffset, fromindex, indexoffset, length)
    outtoindex = [0, 0, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_mask8')
    funcPy(tomask, fromindex, indexoffset, length)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_IndexedArray64_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_mask8')
    funcPy(tomask, fromindex, indexoffset, length)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_IndexedArrayU32_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_mask8')
    funcPy(tomask, fromindex, indexoffset, length)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_ByteMaskedArray_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frommask = [1, 1, 1, 1, 1]
    maskoffset = 0
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_mask8')
    funcPy(tomask, frommask, maskoffset, length, validwhen)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_zero_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_zero_mask8')
    funcPy(tomask, length)
    outtomask = [0, 0, 0]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_IndexedArrayU32_simplifyU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_simplifyU32_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_simplifyU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_simplifyU32_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_simplify32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_simplify32_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_simplify64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_simplify64_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArrayU32_simplify64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_simplify64_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArrayU32_simplify32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_simplify32_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_simplify64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_simplify64_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_simplifyU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_simplifyU32_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_simplify32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outerindex = [1, 0, 0, 1, 1, 1, 0]
    outeroffset = 1
    outerlength = 3
    innerindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    inneroffset = 0
    innerlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_simplify32_to64')
    funcPy(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength)
    outtoindex = [14, 14, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_RegularArray_compact_offsets64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    size = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_compact_offsets64')
    funcPy(tooffsets, length, size)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArray64_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    startsoffset = 0
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_compact_offsets_64')
    funcPy(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArrayU32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    startsoffset = 0
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_compact_offsets_64')
    funcPy(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArray32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    startsoffset = 0
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_compact_offsets_64')
    funcPy(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length)
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListOffsetArray32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray32_compact_offsets_64')
    funcPy(tooffsets, fromoffsets, offsetsoffset, length)
    outtooffsets = [0, 1, 2, 4]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListOffsetArray64_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray64_compact_offsets_64')
    funcPy(tooffsets, fromoffsets, offsetsoffset, length)
    outtooffsets = [0, 1, 2, 4]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListOffsetArrayU32_compact_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArrayU32_compact_offsets_64')
    funcPy(tooffsets, fromoffsets, offsetsoffset, length)
    outtooffsets = [0, 1, 2, 4]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArray32_broadcast_tooffsets_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 2, 3, 4, 5, 6]
    offsetsoffset = 0
    offsetslength = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_broadcast_tooffsets_64')
    funcPy(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent)
    outtocarry = [2.0, 0.0]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArray64_broadcast_tooffsets_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 2, 3, 4, 5, 6]
    offsetsoffset = 0
    offsetslength = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_broadcast_tooffsets_64')
    funcPy(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent)
    outtocarry = [2.0, 0.0]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArrayU32_broadcast_tooffsets_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 2, 3, 4, 5, 6]
    offsetsoffset = 0
    offsetslength = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_broadcast_tooffsets_64')
    funcPy(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent)
    outtocarry = [2.0, 0.0]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_RegularArray_broadcast_tooffsets_64_1():
    fromoffsets = [0, 3, 2, 9, 12, 15]
    offsetsoffset = 0
    offsetslength = 3
    size = 2
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_broadcast_tooffsets_64')
    try:
        funcPy(fromoffsets, offsetsoffset, offsetslength, size)
        assert False
    except:
        pass

def test_awkward_RegularArray_broadcast_tooffsets_size1_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [0, 3, 6, 9, 12, 15]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_broadcast_tooffsets_size1_64')
    funcPy(tocarry, fromoffsets, offsetsoffset, offsetslength)
    outtocarry = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListOffsetArrayU32_toRegularArray_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArrayU32_toRegularArray')
    funcPy(size, fromoffsets, offsetsoffset, offsetslength)
    outsize = [1]
    for i in range(len(outsize)):
        assert size[i] == outsize[i]

def test_awkward_ListOffsetArray32_toRegularArray_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray32_toRegularArray')
    funcPy(size, fromoffsets, offsetsoffset, offsetslength)
    outsize = [1]
    for i in range(len(outsize)):
        assert size[i] == outsize[i]

def test_awkward_ListOffsetArray64_toRegularArray_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    offsetsoffset = 0
    offsetslength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray64_toRegularArray')
    funcPy(size, fromoffsets, offsetsoffset, offsetslength)
    outsize = [1]
    for i in range(len(outsize)):
        assert size[i] == outsize[i]

def test_awkward_NumpyArray_fill_touint64_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint64_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint16_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint16_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint8_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint8_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint32_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint32_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint8_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint8_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint16_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint16_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint32_frombool_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [True, False, True, True, False, True, True, True, True, False, True, True, False, False, False, True, True, False, True, False, True]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint32_frombool')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [1.0, 0.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint8_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint8_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint64_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint64_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint32_fromuint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint32_fromuint32')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint32_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint32_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint32_fromint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint32_fromint32')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint16_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint16_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint16_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint16_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint32_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint32_fromuint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint32_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint32_fromuint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint64_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint64_fromuint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromuint64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromuint64')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint32_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint32_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromint32')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromint64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromint64')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint64_fromuint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint64_fromuint32')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint8_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint8_fromint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint16_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint16_fromuint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint16_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint16_fromint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_touint64_fromuint64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_touint64_fromuint64')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromuint32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromuint32')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromuint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromuint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint64_fromuint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint64_fromuint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint32_fromint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint32_fromint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint16_fromint16_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint16_fromint16')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_fill_toint32_fromint8_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffset = 0
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    fromoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_fill_toint32_fromint8')
    funcPy(toptr, tooffset, fromptr, fromoffset, length)
    outtoptr = [5.0, 8.0, 7.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_ListArray_fill_to64_from64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostartsoffset = 3
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostopsoffset = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstartsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstopsoffset = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray_fill_to64_from64')
    funcPy(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base)
    outtostarts = [0, 0, 0, 5.0, 3.0, 5.0]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [0, 0, 0, 6.0, 4.0, 6.0]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListArray_fill_to64_fromU32_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostartsoffset = 3
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostopsoffset = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstartsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstopsoffset = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray_fill_to64_fromU32')
    funcPy(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base)
    outtostarts = [0, 0, 0, 5.0, 3.0, 5.0]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [0, 0, 0, 6.0, 4.0, 6.0]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListArray_fill_to64_from32_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostartsoffset = 3
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostopsoffset = 3
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstartsoffset = 0
    fromstops = [3, 1, 3, 2, 3]
    fromstopsoffset = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray_fill_to64_from32')
    funcPy(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base)
    outtostarts = [0, 0, 0, 5.0, 3.0, 5.0]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [0, 0, 0, 6.0, 4.0, 6.0]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_IndexedArray_fill_to64_from32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 1
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray_fill_to64_from32')
    funcPy(toindex, toindexoffset, fromindex, fromindexoffset, length, base)
    outtoindex = [0, 0, 0, 3.0, 3.0, 4.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray_fill_to64_fromU32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 1
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray_fill_to64_fromU32')
    funcPy(toindex, toindexoffset, fromindex, fromindexoffset, length, base)
    outtoindex = [0, 0, 0, 3.0, 3.0, 4.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray_fill_to64_from64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 1
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray_fill_to64_from64')
    funcPy(toindex, toindexoffset, fromindex, fromindexoffset, length, base)
    outtoindex = [0, 0, 0, 3.0, 3.0, 4.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray_fill_to64_count_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 1
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray_fill_to64_count')
    funcPy(toindex, toindexoffset, length, base)
    outtoindex = [0, 3, 4, 5]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray_filltags_to8_from8_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totagsoffset = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_filltags_to8_from8')
    funcPy(totags, totagsoffset, fromtags, fromtagsoffset, length, base)
    outtotags = [0, 0, 0, 3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]

def test_awkward_UnionArray_fillindex_to64_from32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillindex_to64_from32')
    funcPy(toindex, toindexoffset, fromindex, fromindexoffset, length)
    outtoindex = [0, 0, 0, 4.0, 3.0, 6.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray_fillindex_to64_from64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillindex_to64_from64')
    funcPy(toindex, toindexoffset, fromindex, fromindexoffset, length)
    outtoindex = [0, 0, 0, 4.0, 3.0, 6.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray_fillindex_to64_fromU32_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillindex_to64_fromU32')
    funcPy(toindex, toindexoffset, fromindex, fromindexoffset, length)
    outtoindex = [0, 0, 0, 4.0, 3.0, 6.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray_filltags_to8_const_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    totagsoffset = 3
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_filltags_to8_const')
    funcPy(totags, totagsoffset, length, base)
    outtotags = [0, 0, 0, 3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]

def test_awkward_UnionArray_fillindex_to64_count_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindexoffset = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillindex_to64_count')
    funcPy(toindex, toindexoffset, length)
    outtoindex = [0, 0, 0, 0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_64_simplify8_64_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_simplify8_64_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_32_simplify8_U32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_simplify8_U32_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_U32_simplify8_64_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_simplify8_64_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_64_simplify8_32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_simplify8_32_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_32_simplify8_32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_simplify8_32_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_U32_simplify8_U32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_simplify8_U32_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_U32_simplify8_32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_simplify8_32_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_32_simplify8_64_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_simplify8_64_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_64_simplify8_U32_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outertagsoffset = 1
    outerindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    outerindexoffset = 0
    innertags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    innertagsoffset = 0
    innerindex = [3, 4, 7, 7, 4, 1, 3, 8, 3, 8, 8]
    innerindexoffset = 1
    towhich = 3
    innerwhich = 1
    outerwhich = 0
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_simplify8_U32_to8_64')
    funcPy(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base)
    outtotags = [3.0, 3.0, 3.0]
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = [4.0, 7.0, 11.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_U32_simplify_one_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    towhich = 3
    fromwhich = 3
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_simplify_one_to8_64')
    funcPy(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base)
    outtotags = []
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = []
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_64_simplify_one_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    towhich = 3
    fromwhich = 3
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_simplify_one_to8_64')
    funcPy(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base)
    outtotags = []
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = []
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_32_simplify_one_to8_64_1():
    totags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtagsoffset = 1
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    fromindexoffset = 0
    towhich = 3
    fromwhich = 3
    length = 3
    base = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_simplify_one_to8_64')
    funcPy(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base)
    outtotags = []
    for i in range(len(outtotags)):
        assert totags[i] == outtotags[i]
    outtoindex = []
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_32_validity_1():
    tags = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    index = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    indexoffset = 0
    length = 3
    numcontents = 3
    lencontents = [7, 8, 9, 10, 11, 12]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_validity')
    try:
        funcPy(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents)
        assert False
    except:
        pass

def test_awkward_UnionArray8_U32_validity_1():
    tags = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    index = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    indexoffset = 0
    length = 3
    numcontents = 3
    lencontents = [7, 8, 9, 10, 11, 12]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_validity')
    try:
        funcPy(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents)
        assert False
    except:
        pass

def test_awkward_UnionArray8_64_validity_1():
    tags = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    index = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    indexoffset = 0
    length = 3
    numcontents = 3
    lencontents = [7, 8, 9, 10, 11, 12]
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_validity')
    try:
        funcPy(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents)
        assert False
    except:
        pass

def test_awkward_UnionArray_fillna_from32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    offset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillna_from32_to64')
    funcPy(toindex, fromindex, offset, length)
    outtoindex = [4, 3, 6]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray_fillna_from64_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    offset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillna_from64_to64')
    funcPy(toindex, fromindex, offset, length)
    outtoindex = [4, 3, 6]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray_fillna_fromU32_to64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [4, 3, 6, 6, 3, 7, 3, 8, 3, 8, 8]
    offset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray_fillna_fromU32_to64')
    funcPy(toindex, fromindex, offset, length)
    outtoindex = [4, 3, 6]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frommask = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64')
    funcPy(toindex, frommask, length)
    outtoindex = [-1, -1, -1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_index_rpad_and_clip_axis0_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_index_rpad_and_clip_axis0_64')
    funcPy(toindex, target, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_index_rpad_and_clip_axis1_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_index_rpad_and_clip_axis1_64')
    funcPy(tostarts, tostops, target, length)
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_RegularArray_rpad_and_clip_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target = 3
    size = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_rpad_and_clip_axis1_64')
    funcPy(toindex, target, size, length)
    outtoindex = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_ListArrayU32_min_range_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_min_range')
    funcPy(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset)
    outtomin = [1]
    for i in range(len(outtomin)):
        assert tomin[i] == outtomin[i]

def test_awkward_ListArray64_min_range_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_min_range')
    funcPy(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset)
    outtomin = [1]
    for i in range(len(outtomin)):
        assert tomin[i] == outtomin[i]

def test_awkward_ListArray32_min_range_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_min_range')
    funcPy(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset)
    outtomin = [1]
    for i in range(len(outtomin)):
        assert tomin[i] == outtomin[i]

def test_awkward_ListArray32_rpad_and_clip_length_axis1_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    target = 3
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_rpad_and_clip_length_axis1')
    funcPy(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset)
    outtomin = [9]
    for i in range(len(outtomin)):
        assert tomin[i] == outtomin[i]

def test_awkward_ListArray64_rpad_and_clip_length_axis1_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    target = 3
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_rpad_and_clip_length_axis1')
    funcPy(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset)
    outtomin = [9]
    for i in range(len(outtomin)):
        assert tomin[i] == outtomin[i]

def test_awkward_ListArrayU32_rpad_and_clip_length_axis1_1():
    tomin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    target = 3
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_rpad_and_clip_length_axis1')
    funcPy(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset)
    outtomin = [9]
    for i in range(len(outtomin)):
        assert tomin[i] == outtomin[i]

def test_awkward_ListArray64_rpad_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target = 3
    length = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_rpad_axis1_64')
    funcPy(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset)
    outtoindex = [2, -1, -1, 0, -1, -1, 2, -1, -1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListArrayU32_rpad_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target = 3
    length = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_rpad_axis1_64')
    funcPy(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset)
    outtoindex = [2, -1, -1, 0, -1, -1, 2, -1, -1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListArray32_rpad_axis1_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target = 3
    length = 3
    startsoffset = 0
    stopsoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_rpad_axis1_64')
    funcPy(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset)
    outtoindex = [2, -1, -1, 0, -1, -1, 2, -1, -1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtostarts = [0, 3, 6]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3, 6, 9]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListOffsetArray64_rpad_length_axis1_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    fromlength = 3
    target = 3
    tolength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray64_rpad_length_axis1')
    funcPy(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtolength = [9]
    for i in range(len(outtolength)):
        assert tolength[i] == outtolength[i]

def test_awkward_ListOffsetArrayU32_rpad_length_axis1_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    fromlength = 3
    target = 3
    tolength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArrayU32_rpad_length_axis1')
    funcPy(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtolength = [9]
    for i in range(len(outtolength)):
        assert tolength[i] == outtolength[i]

def test_awkward_ListOffsetArray32_rpad_length_axis1_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    fromlength = 3
    target = 3
    tolength = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray32_rpad_length_axis1')
    funcPy(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength)
    outtooffsets = [0, 3, 6, 9]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtolength = [9]
    for i in range(len(outtolength)):
        assert tolength[i] == outtolength[i]

def test_awkward_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_localindex_64')
    funcPy(toindex, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_ListArray32_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_localindex_64')
    funcPy(toindex, offsets, offsetsoffset, length)
    outtoindex = [0, 0, 0, 0, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_ListArrayU32_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_localindex_64')
    funcPy(toindex, offsets, offsetsoffset, length)
    outtoindex = [0, 0, 0, 0, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_ListArray64_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_localindex_64')
    funcPy(toindex, offsets, offsetsoffset, length)
    outtoindex = [0, 0, 0, 0, 1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_RegularArray_localindex_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_localindex_64')
    funcPy(toindex, size, length)
    outtoindex = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_ListArray32_combinations_length_64_1():
    totallen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n = 3
    replacement = True
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    stops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_combinations_length_64')
    funcPy(totallen, tooffsets, n, replacement, starts, startsoffset, stops, stopsoffset, length)
    outtotallen = [3]
    for i in range(len(outtotallen)):
        assert totallen[i] == outtotallen[i]
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArray64_combinations_length_64_1():
    totallen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n = 3
    replacement = True
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    stops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_combinations_length_64')
    funcPy(totallen, tooffsets, n, replacement, starts, startsoffset, stops, stopsoffset, length)
    outtotallen = [3]
    for i in range(len(outtotallen)):
        assert totallen[i] == outtotallen[i]
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArrayU32_combinations_length_64_1():
    totallen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n = 3
    replacement = True
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    stops = [3, 1, 3, 2, 3]
    stopsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_combinations_length_64')
    funcPy(totallen, tooffsets, n, replacement, starts, startsoffset, stops, stopsoffset, length)
    outtotallen = [3]
    for i in range(len(outtotallen)):
        assert totallen[i] == outtotallen[i]
    outtooffsets = [0, 1, 2, 3]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ByteMaskedArray_overlay_mask8_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    theirmask = [1, 1, 1, 1, 1]
    theirmaskoffset = 0
    mymask = [1, 1, 1, 1, 1]
    mymaskoffset = 0
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_overlay_mask8')
    funcPy(tomask, theirmask, theirmaskoffset, mymask, mymaskoffset, length, validwhen)
    outtomask = [1, 1, 1]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_BitMaskedArray_to_ByteMaskedArray_1():
    tobytemask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frombitmask = [244, 251, 64, 0, 133]
    bitmaskoffset = 1
    bitmasklength = 3
    validwhen = True
    lsb_order = False
    funcPy = getattr(tests.kernels, 'awkward_BitMaskedArray_to_ByteMaskedArray')
    funcPy(tobytemask, frombitmask, bitmaskoffset, bitmasklength, validwhen, lsb_order)
    outtobytemask = [False, False, False, False, False, True, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    for i in range(len(outtobytemask)):
        assert tobytemask[i] == outtobytemask[i]

def test_awkward_BitMaskedArray_to_IndexedOptionArray64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frombitmask = [244, 251, 64, 0, 133]
    bitmaskoffset = 1
    bitmasklength = 3
    validwhen = True
    lsb_order = False
    funcPy = getattr(tests.kernels, 'awkward_BitMaskedArray_to_IndexedOptionArray64')
    funcPy(toindex, frombitmask, bitmaskoffset, bitmasklength, validwhen, lsb_order)
    outtoindex = [0, 1, 2, 3, 4, -1, 6, 7, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_reduce_count_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    parents = [1, 0, 0, 1, 1, 1, 0]
    parentsoffset = 1
    lenparents = 3
    outlength = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_count_64')
    funcPy(toptr, parents, parentsoffset, lenparents, outlength)
    outtoptr = [2, 1, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_countnonzero_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_countnonzero_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int64_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int64_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint32_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint32_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint32_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint32_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint32_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int32_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int32_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint64_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int32_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint64_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint64_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint64_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint64_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int64_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int64_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_uint64_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_uint64_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int64_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int32_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int32_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int64_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int64_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int64_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int64_bool_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_int32_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_int32_bool_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_int8_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_uint16_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_int16_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_uint64_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_int32_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_uint32_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_int64_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_sum_bool_uint8_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_sum_bool_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int32_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int32_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int64_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint32_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint32_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint32_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint32_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint32_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int64_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int64_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint64_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint64_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint64_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint64_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint64_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int64_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int64_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int32_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_uint64_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_uint64_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int32_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int32_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int64_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int64_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int64_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int64_bool_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_int32_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_int32_bool_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_uint64_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_uint16_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_uint32_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_uint8_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_int32_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_int16_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_int8_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_prod_bool_int64_64_1():
    toptr = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_prod_bool_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_int8_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_int8_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_uint16_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_uint16_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_int16_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_int16_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_uint32_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_uint8_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_uint8_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_int64_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_int32_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_min_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_min_uint64_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_int64_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_int64_int64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_int8_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_int8_int8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_int32_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_int32_int32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_uint8_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_uint8_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_int16_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_int16_int16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_uint64_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_uint64_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_uint32_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_uint32_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_max_uint16_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    identity = 3
    funcPy = getattr(tests.kernels, 'awkward_reduce_max_uint16_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity)
    outtoptr = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_int32_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_int64_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_int8_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_int16_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmin_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmin_bool_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_uint8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_uint8_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_uint64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_uint64_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_uint32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_uint32_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_int32_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_int32_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_uint16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_uint16_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_int64_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_int64_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_int16_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_int16_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_int8_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_int8_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_reduce_argmax_bool_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [True, False, True, False, True, True, False, True, False, True]
    fromptroffset = 1
    starts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    startsoffset = 0
    parents = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    parentsoffset = 0
    lenparents = 3
    outlength = 30
    funcPy = getattr(tests.kernels, 'awkward_reduce_argmax_bool_64')
    funcPy(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength)
    outtoptr = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_content_reduce_zeroparents_64_1():
    toparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_content_reduce_zeroparents_64')
    funcPy(toparents, length)
    outtoparents = [0, 0, 0]
    for i in range(len(outtoparents)):
        assert toparents[i] == outtoparents[i]

def test_awkward_ListOffsetArray_reduce_global_startstop_64_1():
    globalstart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    globalstop = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_global_startstop_64')
    funcPy(globalstart, globalstop, offsets, offsetsoffset, length)
    outglobalstart = [1]
    for i in range(len(outglobalstart)):
        assert globalstart[i] == outglobalstart[i]
    outglobalstop = [5]
    for i in range(len(outglobalstop)):
        assert globalstop[i] == outglobalstop[i]

def test_awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_1():
    maxcount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsetscopy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64')
    funcPy(maxcount, offsetscopy, offsets, offsetsoffset, length)
    outmaxcount = [2]
    for i in range(len(outmaxcount)):
        assert maxcount[i] == outmaxcount[i]
    outoffsetscopy = [1, 2, 3, 5]
    for i in range(len(outoffsetscopy)):
        assert offsetscopy[i] == outoffsetscopy[i]

def test_awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_1():
    nextstarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    nextlen = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64')
    funcPy(nextstarts, nextparents, nextlen)
    outnextstarts = [0, 0, 2]
    for i in range(len(outnextstarts)):
        assert nextstarts[i] == outnextstarts[i]

def test_awkward_ListOffsetArray_reduce_nonlocal_findgaps_64_1():
    gaps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    parents = [1, 0, 0, 1, 1, 1, 0]
    parentsoffset = 1
    lenparents = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_nonlocal_findgaps_64')
    funcPy(gaps, parents, parentsoffset, lenparents)
    outgaps = [1, 1]
    for i in range(len(outgaps)):
        assert gaps[i] == outgaps[i]

def test_awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_1():
    outstarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outstops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    distincts = [1, 0, 0, 1, 1, 1, 0]
    lendistincts = 3
    gaps = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    outlength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64')
    funcPy(outstarts, outstops, distincts, lendistincts, gaps, outlength)
    outoutstarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(outoutstarts)):
        assert outstarts[i] == outoutstarts[i]
    outoutstops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    for i in range(len(outoutstops)):
        assert outstops[i] == outoutstops[i]

def test_awkward_ListOffsetArray_reduce_local_nextparents_64_1():
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    offsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_local_nextparents_64')
    funcPy(nextparents, offsets, offsetsoffset, length)
    outnextparents = [0, 1, 2, 2]
    for i in range(len(outnextparents)):
        assert nextparents[i] == outnextparents[i]

def test_awkward_ListOffsetArray_reduce_local_outoffsets_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    parents = [1, 0, 0, 1, 1, 1, 0]
    parentsoffset = 1
    lenparents = 3
    outlength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_reduce_local_outoffsets_64')
    funcPy(outoffsets, parents, parentsoffset, lenparents, outlength)
    outoutoffsets = [0, 2, 3, 3]
    for i in range(len(outoutoffsets)):
        assert outoffsets[i] == outoutoffsets[i]

def test_awkward_IndexedArray64_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parentsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_reduce_next_64')
    funcPy(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length)
    outnextcarry = [0, 0, 1]
    for i in range(len(outnextcarry)):
        assert nextcarry[i] == outnextcarry[i]
    outnextparents = [1, 2, 3]
    for i in range(len(outnextparents)):
        assert nextparents[i] == outnextparents[i]
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert outindex[i] == outoutindex[i]

def test_awkward_IndexedArrayU32_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parentsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_reduce_next_64')
    funcPy(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length)
    outnextcarry = [0, 0, 1]
    for i in range(len(outnextcarry)):
        assert nextcarry[i] == outnextcarry[i]
    outnextparents = [1, 2, 3]
    for i in range(len(outnextparents)):
        assert nextparents[i] == outnextparents[i]
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert outindex[i] == outoutindex[i]

def test_awkward_IndexedArray32_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parentsoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_reduce_next_64')
    funcPy(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length)
    outnextcarry = [0, 0, 1]
    for i in range(len(outnextcarry)):
        assert nextcarry[i] == outnextcarry[i]
    outnextparents = [1, 2, 3]
    for i in range(len(outnextparents)):
        assert nextparents[i] == outnextparents[i]
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert outindex[i] == outoutindex[i]

def test_awkward_IndexedArray_reduce_next_fix_offsets_64_1():
    outoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    starts = [1, 0, 0, 1, 1, 1, 0]
    startsoffset = 1
    startslength = 3
    outindexlength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray_reduce_next_fix_offsets_64')
    funcPy(outoffsets, starts, startsoffset, startslength, outindexlength)
    outoutoffsets = [0, 0, 1, 0, 3]
    for i in range(len(outoutoffsets)):
        assert outoffsets[i] == outoutoffsets[i]

def test_awkward_NumpyArray_reduce_mask_ByteMaskedArray_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    parents = [1, 0, 0, 1, 1, 1, 0]
    parentsoffset = 1
    lenparents = 3
    outlength = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_reduce_mask_ByteMaskedArray_64')
    funcPy(toptr, parents, parentsoffset, lenparents, outlength)
    outtoptr = [0, 0, 1]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_ByteMaskedArray_reduce_next_64_1():
    nextcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextparents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [1, 1, 1, 1, 1]
    maskoffset = 0
    parents = [1, 0, 0, 1, 1, 1, 0]
    parentsoffset = 1
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_reduce_next_64')
    funcPy(nextcarry, nextparents, outindex, mask, maskoffset, parents, parentsoffset, length, validwhen)
    outnextcarry = [0, 1, 2]
    for i in range(len(outnextcarry)):
        assert nextcarry[i] == outnextcarry[i]
    outnextparents = [0, 0, 1]
    for i in range(len(outnextparents)):
        assert nextparents[i] == outnextparents[i]
    outoutindex = [0, 1, 2]
    for i in range(len(outoutindex)):
        assert outindex[i] == outoutindex[i]

def test_awkward_Index8_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index8_to_Index64')
    funcPy(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_IndexU8_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexU8_to_Index64')
    funcPy(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Index32_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index32_to_Index64')
    funcPy(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_IndexU32_to_Index64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [1, 0, 0, 1, 1, 1, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexU32_to_Index64')
    funcPy(toptr, fromptr, length)
    outtoptr = [1, 0, 0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_IndexU8_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexU8_carry_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Index64_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index64_carry_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexU32_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexU32_carry_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Index8_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index8_carry_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Index32_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    lenfromindex = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index32_carry_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, lenfromindex, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexU32_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexU32_carry_nocheck_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Index32_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index32_carry_nocheck_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexU8_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexU8_carry_nocheck_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Index8_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index8_carry_nocheck_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Index64_carry_nocheck_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    carry = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Index64_carry_nocheck_64')
    funcPy(toindex, fromindex, carry, fromindexoffset, length)
    outtoindex = [1, 14, 14]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_slicemissing_check_same_1():
    same = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    bytemask = [1, 1, 1, 1, 1]
    bytemaskoffset = 0
    missingindex = [1, 1, 1, 1, 1]
    missingindexoffset = 0
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_slicemissing_check_same')
    funcPy(same, bytemask, bytemaskoffset, missingindex, missingindexoffset, length)
    outsame = [False]
    for i in range(len(outsame)):
        assert same[i] == outsame[i]

def test_awkward_carry_arange32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_carry_arange32')
    funcPy(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_carry_arange64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_carry_arange64')
    funcPy(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_carry_arangeU32_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_carry_arangeU32')
    funcPy(toptr, length)
    outtoptr = [0, 1, 2]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_Identities64_getitem_carry_64_1():
    newidentitiesptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    identitiesptr = [1, 0, 0, 1, 1, 1, 0]
    carryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lencarry = 3
    offset = 1
    width = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities64_getitem_carry_64')
    funcPy(newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length)
    outnewidentitiesptr = [0, 0, 1, 0, 0, 1, 0, 0, 1]
    for i in range(len(outnewidentitiesptr)):
        assert newidentitiesptr[i] == outnewidentitiesptr[i]

def test_awkward_Identities32_getitem_carry_64_1():
    newidentitiesptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    identitiesptr = [1, 0, 0, 1, 1, 1, 0]
    carryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lencarry = 3
    offset = 1
    width = 3
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Identities32_getitem_carry_64')
    funcPy(newidentitiesptr, identitiesptr, carryptr, lencarry, offset, width, length)
    outnewidentitiesptr = [0, 0, 1, 0, 0, 1, 0, 0, 1]
    for i in range(len(outnewidentitiesptr)):
        assert newidentitiesptr[i] == outnewidentitiesptr[i]

def test_awkward_NumpyArray_contiguous_init_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    skip = 3
    stride = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_contiguous_init_64')
    funcPy(toptr, skip, stride)
    outtoptr = [0, 3, 6]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_NumpyArray_contiguous_next_64_1():
    topos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frompos = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    length = 3
    skip = 3
    stride = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_contiguous_next_64')
    funcPy(topos, frompos, length, skip, stride)
    outtopos = [5, 8, 11, 8, 11, 14, 7, 10, 13]
    for i in range(len(outtopos)):
        assert topos[i] == outtopos[i]

def test_awkward_NumpyArray_getitem_next_at_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    lencarry = 3
    skip = 3
    at = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_next_at_64')
    funcPy(nextcarryptr, carryptr, lencarry, skip, at)
    outnextcarryptr = [18, 27, 24]
    for i in range(len(outnextcarryptr)):
        assert nextcarryptr[i] == outnextcarryptr[i]

def test_awkward_NumpyArray_getitem_next_range_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    lencarry = 3
    lenhead = 3
    skip = 3
    start = 3
    step = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_next_range_64')
    funcPy(nextcarryptr, carryptr, lencarry, lenhead, skip, start, step)
    outnextcarryptr = [18, 21, 24, 27, 30, 33, 24, 27, 30]
    for i in range(len(outnextcarryptr)):
        assert nextcarryptr[i] == outnextcarryptr[i]

def test_awkward_NumpyArray_getitem_next_range_advanced_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextadvancedptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    advancedptr = [2, 1, 4, 3, 5, 2, 10, 11, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 6, 2, 3]
    lencarry = 3
    lenhead = 3
    skip = 3
    start = 3
    step = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_next_range_advanced_64')
    funcPy(nextcarryptr, nextadvancedptr, carryptr, advancedptr, lencarry, lenhead, skip, start, step)
    outnextcarryptr = [18, 21, 24, 27, 30, 33, 24, 27, 30]
    for i in range(len(outnextcarryptr)):
        assert nextcarryptr[i] == outnextcarryptr[i]
    outnextadvancedptr = [2, 2, 2, 1, 1, 1, 4, 4, 4]
    for i in range(len(outnextadvancedptr)):
        assert nextadvancedptr[i] == outnextadvancedptr[i]

def test_awkward_NumpyArray_getitem_next_array_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nextadvancedptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    flatheadptr = [2, 1, 4, 3, 5, 2, 10, 11, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 6, 2, 3]
    lencarry = 3
    lenflathead = 3
    skip = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_next_array_64')
    funcPy(nextcarryptr, nextadvancedptr, carryptr, flatheadptr, lencarry, lenflathead, skip)
    outnextcarryptr = [17, 16, 19, 26, 25, 28, 23, 22, 25]
    for i in range(len(outnextcarryptr)):
        assert nextcarryptr[i] == outnextcarryptr[i]
    outnextadvancedptr = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outnextadvancedptr)):
        assert nextadvancedptr[i] == outnextadvancedptr[i]

def test_awkward_NumpyArray_getitem_next_array_advanced_64_1():
    nextcarryptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carryptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    advancedptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    flatheadptr = [2, 1, 4, 3, 5, 2, 10, 11, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 6, 2, 3]
    lencarry = 3
    skip = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_next_array_advanced_64')
    funcPy(nextcarryptr, carryptr, advancedptr, flatheadptr, lencarry, skip)
    outnextcarryptr = [17, 44, 32]
    for i in range(len(outnextcarryptr)):
        assert nextcarryptr[i] == outnextcarryptr[i]

def test_awkward_NumpyArray_getitem_boolean_numtrue_1():
    numtrue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    byteoffset = 0
    length = 3
    stride = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_boolean_numtrue')
    funcPy(numtrue, fromptr, byteoffset, length, stride)
    outnumtrue = [1]
    for i in range(len(outnumtrue)):
        assert numtrue[i] == outnumtrue[i]

def test_awkward_NumpyArray_getitem_boolean_nonzero_64_1():
    toptr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromptr = [5, 8, 7, 0, 8, 6, 12, 7, 4, 6, 8, 7, 8, 4, 4, 4, 9, 10, 6, 5, 3, 5, 7, 5, 3]
    byteoffset = 0
    length = 3
    stride = 3
    funcPy = getattr(tests.kernels, 'awkward_NumpyArray_getitem_boolean_nonzero_64')
    funcPy(toptr, fromptr, byteoffset, length, stride)
    outtoptr = [0]
    for i in range(len(outtoptr)):
        assert toptr[i] == outtoptr[i]

def test_awkward_ListArray32_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    at = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_next_at_64')
    funcPy(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArray64_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    at = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_next_at_64')
    funcPy(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArrayU32_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    lenstarts = 3
    startsoffset = 0
    stopsoffset = 0
    at = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_next_at_64')
    funcPy(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset, at)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArrayU32_getitem_next_range_counts_64_1():
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    lenstarts = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_next_range_counts_64')
    funcPy(total, fromoffsets, lenstarts)
    outtotal = [-1]
    for i in range(len(outtotal)):
        assert total[i] == outtotal[i]

def test_awkward_ListArray64_getitem_next_range_counts_64_1():
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    lenstarts = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_next_range_counts_64')
    funcPy(total, fromoffsets, lenstarts)
    outtotal = [-1]
    for i in range(len(outtotal)):
        assert total[i] == outtotal[i]

def test_awkward_ListArray32_getitem_next_range_counts_64_1():
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    lenstarts = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_next_range_counts_64')
    funcPy(total, fromoffsets, lenstarts)
    outtotal = [-1]
    for i in range(len(outtotal)):
        assert total[i] == outtotal[i]

def test_awkward_ListArray32_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromoffsets = [1, 2, 3, 4, 5, 6]
    lenstarts = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_next_range_spreadadvanced_64')
    funcPy(toadvanced, fromadvanced, fromoffsets, lenstarts)
    outtoadvanced = [0, 2, 0, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArray64_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromoffsets = [1, 2, 3, 4, 5, 6]
    lenstarts = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_next_range_spreadadvanced_64')
    funcPy(toadvanced, fromadvanced, fromoffsets, lenstarts)
    outtoadvanced = [0, 2, 0, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArrayU32_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromoffsets = [1, 2, 3, 4, 5, 6]
    lenstarts = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_next_range_spreadadvanced_64')
    funcPy(toadvanced, fromadvanced, fromoffsets, lenstarts)
    outtoadvanced = [0, 2, 0, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArrayU32_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_next_array_64')
    funcPy(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 2, 2, 0, 0, 0, 2, 2, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArray32_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_next_array_64')
    funcPy(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 2, 2, 0, 0, 0, 2, 2, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArray64_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_next_array_64')
    funcPy(tocarry, toadvanced, fromstarts, fromstops, fromarray, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 2, 2, 0, 0, 0, 2, 2, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArrayU32_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [1, 2, 3, 4, 5, 6]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_next_array_advanced_64')
    funcPy(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArray32_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [1, 2, 3, 4, 5, 6]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_next_array_advanced_64')
    funcPy(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArray64_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [1, 2, 3, 4, 5, 6]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lenarray = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_next_array_advanced_64')
    funcPy(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, startsoffset, stopsoffset, lenstarts, lenarray, lencontent)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_ListArrayU32_getitem_carry_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_carry_64')
    funcPy(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry)
    outtostarts = [2.0, 2.0, 2.0]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3.0, 3.0, 3.0]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListArray64_getitem_carry_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_carry_64')
    funcPy(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry)
    outtostarts = [2.0, 2.0, 2.0]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3.0, 3.0, 3.0]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_ListArray32_getitem_carry_64_1():
    tostarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tostops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    fromstops = [3, 1, 3, 2, 3]
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startsoffset = 0
    stopsoffset = 0
    lenstarts = 3
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_carry_64')
    funcPy(tostarts, tostops, fromstarts, fromstops, fromcarry, startsoffset, stopsoffset, lenstarts, lencarry)
    outtostarts = [2.0, 2.0, 2.0]
    for i in range(len(outtostarts)):
        assert tostarts[i] == outtostarts[i]
    outtostops = [3.0, 3.0, 3.0]
    for i in range(len(outtostops)):
        assert tostops[i] == outtostops[i]

def test_awkward_RegularArray_getitem_next_at_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    at = 0
    length = 3
    size = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_next_at_64')
    funcPy(tocarry, at, length, size)
    outtocarry = [0, 3, 6]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_RegularArray_getitem_next_range_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    regular_start = 3
    step = 3
    length = 3
    size = 3
    nextsize = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_next_range_64')
    funcPy(tocarry, regular_start, step, length, size, nextsize)
    outtocarry = [3, 6, 9, 6, 9, 12, 9, 12, 15]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_RegularArray_getitem_next_range_spreadadvanced_64_1():
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [0, 3, 6, 9, 12, 15]
    length = 3
    nextsize = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_next_range_spreadadvanced_64')
    funcPy(toadvanced, fromadvanced, length, nextsize)
    outtoadvanced = [0, 0, 0, 3, 3, 3, 6, 6, 6]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_RegularArray_getitem_next_array_regularize_64_1():
    toarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lenarray = 3
    size = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_next_array_regularize_64')
    funcPy(toarray, fromarray, lenarray, size)
    outtoarray = [0, 0, 0]
    for i in range(len(outtoarray)):
        assert toarray[i] == outtoarray[i]

def test_awkward_RegularArray_getitem_next_array_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    lenarray = 3
    size = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_next_array_64')
    funcPy(tocarry, toadvanced, fromarray, length, lenarray, size)
    outtocarry = [0, 0, 0, 3, 3, 3, 6, 6, 6]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_RegularArray_getitem_next_array_advanced_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toadvanced = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromadvanced = [0, 3, 6, 9, 12, 15]
    fromarray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    lenarray = 3
    size = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_next_array_advanced_64')
    funcPy(tocarry, toadvanced, fromadvanced, fromarray, length, lenarray, size)
    outtocarry = [0, 3, 6]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoadvanced = [0, 1, 2]
    for i in range(len(outtoadvanced)):
        assert toadvanced[i] == outtoadvanced[i]

def test_awkward_RegularArray_getitem_carry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromcarry = [0, 3, 6, 9, 12, 15]
    lencarry = 3
    size = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_carry_64')
    funcPy(tocarry, fromcarry, lencarry, size)
    outtocarry = [0, 1, 2, 9, 10, 11, 18, 19, 20]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArray64_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_numnull')
    funcPy(numnull, fromindex, indexoffset, lenindex)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert numnull[i] == outnumnull[i]

def test_awkward_IndexedArrayU32_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_numnull')
    funcPy(numnull, fromindex, indexoffset, lenindex)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert numnull[i] == outnumnull[i]

def test_awkward_IndexedArray32_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_numnull')
    funcPy(numnull, fromindex, indexoffset, lenindex)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert numnull[i] == outnumnull[i]

def test_awkward_IndexedArray32_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_getitem_nextcarry_outindex_64')
    funcPy(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_getitem_nextcarry_outindex_64')
    funcPy(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArrayU32_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_getitem_nextcarry_outindex_64')
    funcPy(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64')
    funcPy(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64')
    funcPy(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64')
    funcPy(tocarry, toindex, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtoindex = [0.0, 1.0, 2.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_ListOffsetArray_getitem_adjust_offsets_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tononzero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    nonzero = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    nonzerooffset = 0
    nonzerolength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_getitem_adjust_offsets_64')
    funcPy(tooffsets, tononzero, fromoffsets, offsetsoffset, length, nonzero, nonzerooffset, nonzerolength)
    outtooffsets = [1, 1, 1, 1]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtononzero = []
    for i in range(len(outtononzero)):
        assert tononzero[i] == outtononzero[i]

def test_awkward_ListOffsetArray_getitem_adjust_offsets_index_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tononzero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromoffsets = [1, 1, 2, 3, 5, 8, 11, 11, 11, 11, 20]
    offsetsoffset = 1
    length = 3
    index = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    indexlength = 3
    nonzero = [48, 50, 51, 55, 55, 55, 58, 66, 81, 92, 92, 97, 101, 113, 116, 126, 148, 153, 172, 194, 201, 204, 214, 231, 233, 248, 252, 253, 253, 263, 289, 301, 325]
    nonzerooffset = 0
    nonzerolength = 3
    originalmask = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    maskoffset = 0
    masklength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListOffsetArray_getitem_adjust_offsets_index_64')
    funcPy(tooffsets, tononzero, fromoffsets, offsetsoffset, length, index, indexoffset, indexlength, nonzero, nonzerooffset, nonzerolength, originalmask, maskoffset, masklength)
    outtooffsets = [1, 1, 1, 1]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtononzero = []
    for i in range(len(outtononzero)):
        assert tononzero[i] == outtononzero[i]

def test_awkward_IndexedArray_getitem_adjust_outindex_64_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tononzero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromindexoffset = 1
    fromindexlength = 3
    nonzero = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    nonzerooffset = 0
    nonzerolength = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray_getitem_adjust_outindex_64')
    funcPy(tomask, toindex, tononzero, fromindex, fromindexoffset, fromindexlength, nonzero, nonzerooffset, nonzerolength)
    outtomask = [False, False, False]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]
    outtoindex = []
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outtononzero = []
    for i in range(len(outtononzero)):
        assert tononzero[i] == outtononzero[i]

def test_awkward_IndexedArray32_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_getitem_nextcarry_64')
    funcPy(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArray64_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_getitem_nextcarry_64')
    funcPy(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArrayU32_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    lenindex = 3
    lencontent = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_getitem_nextcarry_64')
    funcPy(tocarry, fromindex, indexoffset, lenindex, lencontent)
    outtocarry = [0, 0, 1]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_IndexedArrayU32_getitem_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexoffset = 1
    lenindex = 3
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArrayU32_getitem_carry_64')
    funcPy(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry)
    outtoindex = [0.0, 0.0, 0.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray32_getitem_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexoffset = 1
    lenindex = 3
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray32_getitem_carry_64')
    funcPy(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry)
    outtoindex = [0.0, 0.0, 0.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_IndexedArray64_getitem_carry_64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    fromcarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexoffset = 1
    lenindex = 3
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_IndexedArray64_getitem_carry_64')
    funcPy(toindex, fromindex, fromcarry, indexoffset, lenindex, lencarry)
    outtoindex = [0.0, 0.0, 0.0]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_UnionArray8_regular_index_getsize_1():
    size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_regular_index_getsize')
    funcPy(size, fromtags, tagsoffset, length)
    outsize = [1]
    for i in range(len(outsize)):
        assert size[i] == outsize[i]

def test_awkward_UnionArray8_32_regular_index_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    current = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_regular_index')
    funcPy(toindex, current, size, fromtags, tagsoffset, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outcurrent = [3, 0, 0]
    for i in range(len(outcurrent)):
        assert current[i] == outcurrent[i]

def test_awkward_UnionArray8_U32_regular_index_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    current = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_regular_index')
    funcPy(toindex, current, size, fromtags, tagsoffset, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outcurrent = [3, 0, 0]
    for i in range(len(outcurrent)):
        assert current[i] == outcurrent[i]

def test_awkward_UnionArray8_64_regular_index_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    current = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    size = 3
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_regular_index')
    funcPy(toindex, current, size, fromtags, tagsoffset, length)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]
    outcurrent = [3, 0, 0]
    for i in range(len(outcurrent)):
        assert current[i] == outcurrent[i]

def test_awkward_UnionArray8_32_project_64_1():
    lenout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    which = 1
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_32_project_64')
    funcPy(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which)
    outlenout = [0]
    for i in range(len(outlenout)):
        assert lenout[i] == outlenout[i]
    outtocarry = []
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_UnionArray8_U32_project_64_1():
    lenout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    which = 1
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_U32_project_64')
    funcPy(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which)
    outlenout = [0]
    for i in range(len(outlenout)):
        assert lenout[i] == outlenout[i]
    outtocarry = []
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_UnionArray8_64_project_64_1():
    lenout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromtags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tagsoffset = 1
    fromindex = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    length = 3
    which = 1
    funcPy = getattr(tests.kernels, 'awkward_UnionArray8_64_project_64')
    funcPy(lenout, tocarry, fromtags, tagsoffset, fromindex, indexoffset, length, which)
    outlenout = [0]
    for i in range(len(outlenout)):
        assert lenout[i] == outlenout[i]
    outtocarry = []
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_missing_repeat_64_1():
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = [1, 0, 0, 1, 1, 1, 0]
    indexoffset = 1
    indexlength = 3
    repetitions = 3
    regularsize = 3
    funcPy = getattr(tests.kernels, 'awkward_missing_repeat_64')
    funcPy(outindex, index, indexoffset, indexlength, repetitions, regularsize)
    outoutindex = [0, 0, 1, 3, 3, 4, 6, 6, 7]
    for i in range(len(outoutindex)):
        assert outindex[i] == outoutindex[i]

def test_awkward_RegularArray_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    singleoffsets = [0, 3, 6, 9, 12, 15]
    regularsize = 3
    regularlength = 3
    funcPy = getattr(tests.kernels, 'awkward_RegularArray_getitem_jagged_expand_64')
    funcPy(multistarts, multistops, singleoffsets, regularsize, regularlength)
    outmultistarts = [0, 3, 6, 0, 3, 6, 0, 3, 6]
    for i in range(len(outmultistarts)):
        assert multistarts[i] == outmultistarts[i]
    outmultistops = [3, 6, 9, 3, 6, 9, 3, 6, 9]
    for i in range(len(outmultistops)):
        assert multistops[i] == outmultistops[i]

def test_awkward_ListArray32_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    singleoffsets = [1, 2, 3, 4, 5, 6]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstopsoffset = 0
    jaggedsize = 2
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_jagged_expand_64')
    funcPy(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, jaggedsize, length)
    outmultistarts = [1, 2, 1, 2, 1, 2]
    for i in range(len(outmultistarts)):
        assert multistarts[i] == outmultistarts[i]
    outmultistops = [2, 3, 2, 3, 2, 3]
    for i in range(len(outmultistops)):
        assert multistops[i] == outmultistops[i]
    outtocarry = [1, 2, 2, 3, 3, 4]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArrayU32_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    singleoffsets = [1, 2, 3, 4, 5, 6]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstopsoffset = 0
    jaggedsize = 2
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_jagged_expand_64')
    funcPy(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, jaggedsize, length)
    outmultistarts = [1, 2, 1, 2, 1, 2]
    for i in range(len(outmultistarts)):
        assert multistarts[i] == outmultistarts[i]
    outmultistops = [2, 3, 2, 3, 2, 3]
    for i in range(len(outmultistops)):
        assert multistops[i] == outmultistops[i]
    outtocarry = [1, 2, 2, 3, 3, 4]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArray64_getitem_jagged_expand_64_1():
    multistarts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    multistops = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    singleoffsets = [1, 2, 3, 4, 5, 6]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstopsoffset = 0
    jaggedsize = 2
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_jagged_expand_64')
    funcPy(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, jaggedsize, length)
    outmultistarts = [1, 2, 1, 2, 1, 2]
    for i in range(len(outmultistarts)):
        assert multistarts[i] == outmultistarts[i]
    outmultistops = [2, 3, 2, 3, 2, 3]
    for i in range(len(outmultistops)):
        assert multistops[i] == outmultistops[i]
    outtocarry = [1, 2, 2, 3, 3, 4]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArray_getitem_jagged_carrylen_64_1():
    carrylen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray_getitem_jagged_carrylen_64')
    funcPy(carrylen, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen)
    outcarrylen = [3]
    for i in range(len(outcarrylen)):
        assert carrylen[i] == outcarrylen[i]

def test_awkward_ListArrayU32_getitem_jagged_apply_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstopsoffset = 0
    contentlen = 10
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_jagged_apply_64')
    funcPy(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
    outtooffsets = [0.0, 1.0, 2.0, 3.0]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtocarry = [1, 2, 3]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArrayU32_getitem_jagged_apply_64_2():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [3, 0, 3, 2, 3]
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [2, 3, 4, 5, 6, 7]
    fromstopsoffset = 0
    contentlen = 4
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_jagged_apply_64')
    try:
        funcPy(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
        assert False
    except:
        pass

def test_awkward_ListArray64_getitem_jagged_apply_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstopsoffset = 0
    contentlen = 10
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_jagged_apply_64')
    funcPy(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
    outtooffsets = [0.0, 1.0, 2.0, 3.0]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtocarry = [1, 2, 3]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArray64_getitem_jagged_apply_64_2():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [3, 0, 3, 2, 3]
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [2, 3, 4, 5, 6, 7]
    fromstopsoffset = 0
    contentlen = 4
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_jagged_apply_64')
    try:
        funcPy(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
        assert False
    except:
        pass

def test_awkward_ListArray32_getitem_jagged_apply_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [3, 4, 5, 6, 7, 8]
    fromstopsoffset = 0
    contentlen = 10
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_jagged_apply_64')
    funcPy(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
    outtooffsets = [0.0, 1.0, 2.0, 3.0]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]
    outtocarry = [1, 2, 3]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ListArray32_getitem_jagged_apply_64_2():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [3, 0, 3, 2, 3]
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestopsoffset = 0
    sliceouterlen = 3
    sliceindex = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sliceindexoffset = 0
    sliceinnerlen = 3
    fromstarts = [1, 2, 3, 4, 5, 6]
    fromstartsoffset = 0
    fromstops = [2, 3, 4, 5, 6, 7]
    fromstopsoffset = 0
    contentlen = 4
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_jagged_apply_64')
    try:
        funcPy(tooffsets, tocarry, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, sliceindex, sliceindexoffset, sliceinnerlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, contentlen)
        assert False
    except:
        pass

def test_awkward_ListArray_getitem_jagged_numvalid_64_1():
    numvalid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestopsoffset = 0
    length = 3
    missing = [1, 2, 3, 4, 5, 6]
    missingoffset = 0
    missinglength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray_getitem_jagged_numvalid_64')
    funcPy(numvalid, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, length, missing, missingoffset, missinglength)
    outnumvalid = [3]
    for i in range(len(outnumvalid)):
        assert numvalid[i] == outnumvalid[i]

def test_awkward_ListArray_getitem_jagged_numvalid_64_2():
    numvalid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [3, 0, 3, 2, 3]
    slicestartsoffset = 0
    slicestops = [2, 10, 3, 1, 3]
    slicestopsoffset = 0
    length = 3
    missing = [0, 4, 1, 2, 3, 3]
    missingoffset = 0
    missinglength = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray_getitem_jagged_numvalid_64')
    try:
        funcPy(numvalid, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, length, missing, missingoffset, missinglength)
        assert False
    except:
        pass

def test_awkward_ListArray_getitem_jagged_shrink_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tosmalloffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tolargeoffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    slicestartsoffset = 0
    slicestops = [3, 1, 3, 2, 3]
    slicestopsoffset = 0
    length = 3
    missing = [1, 2, 3, 4, 5, 6]
    missingoffset = 0
    funcPy = getattr(tests.kernels, 'awkward_ListArray_getitem_jagged_shrink_64')
    funcPy(tocarry, tosmalloffsets, tolargeoffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, length, missing, missingoffset)
    outtocarry = [2, 0, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outtosmalloffsets = [2, 3, 4, 5]
    for i in range(len(outtosmalloffsets)):
        assert tosmalloffsets[i] == outtosmalloffsets[i]
    outtolargeoffsets = [2, 3, 4, 5]
    for i in range(len(outtolargeoffsets)):
        assert tolargeoffsets[i] == outtolargeoffsets[i]

def test_awkward_ListArray32_getitem_jagged_descend_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [1, 2, 3, 4, 5, 6]
    slicestartsoffset = 0
    slicestops = [3, 4, 5, 6, 7, 8]
    slicestopsoffset = 0
    sliceouterlen = 3
    fromstarts = [2, 3, 4, 5, 6, 7]
    fromstartsoffset = 3
    fromstops = [4, 5, 6, 7, 8, 9]
    fromstopsoffset = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray32_getitem_jagged_descend_64')
    funcPy(tooffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset)
    outtooffsets = [1, 3.0, 5.0, 7.0]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArrayU32_getitem_jagged_descend_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [1, 2, 3, 4, 5, 6]
    slicestartsoffset = 0
    slicestops = [3, 4, 5, 6, 7, 8]
    slicestopsoffset = 0
    sliceouterlen = 3
    fromstarts = [2, 3, 4, 5, 6, 7]
    fromstartsoffset = 3
    fromstops = [4, 5, 6, 7, 8, 9]
    fromstopsoffset = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArrayU32_getitem_jagged_descend_64')
    funcPy(tooffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset)
    outtooffsets = [1, 3.0, 5.0, 7.0]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ListArray64_getitem_jagged_descend_64_1():
    tooffsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    slicestarts = [1, 2, 3, 4, 5, 6]
    slicestartsoffset = 0
    slicestops = [3, 4, 5, 6, 7, 8]
    slicestopsoffset = 0
    sliceouterlen = 3
    fromstarts = [2, 3, 4, 5, 6, 7]
    fromstartsoffset = 3
    fromstops = [4, 5, 6, 7, 8, 9]
    fromstopsoffset = 3
    funcPy = getattr(tests.kernels, 'awkward_ListArray64_getitem_jagged_descend_64')
    funcPy(tooffsets, slicestarts, slicestartsoffset, slicestops, slicestopsoffset, sliceouterlen, fromstarts, fromstartsoffset, fromstops, fromstopsoffset)
    outtooffsets = [1, 3.0, 5.0, 7.0]
    for i in range(len(outtooffsets)):
        assert tooffsets[i] == outtooffsets[i]

def test_awkward_ByteMaskedArray_getitem_carry_64_1():
    tomask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frommask = [1, 1, 1, 1, 1]
    frommaskoffset = 0
    lenmask = 3
    fromcarry = [1, 0, 0, 1, 1, 1, 0]
    lencarry = 3
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_getitem_carry_64')
    funcPy(tomask, frommask, frommaskoffset, lenmask, fromcarry, lencarry)
    outtomask = [1, 1, 1]
    for i in range(len(outtomask)):
        assert tomask[i] == outtomask[i]

def test_awkward_ByteMaskedArray_numnull_1():
    numnull = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [1, 1, 1, 1, 1]
    maskoffset = 0
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_numnull')
    funcPy(numnull, mask, maskoffset, length, validwhen)
    outnumnull = [0]
    for i in range(len(outnumnull)):
        assert numnull[i] == outnumnull[i]

def test_awkward_ByteMaskedArray_getitem_nextcarry_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [1, 1, 1, 1, 1]
    maskoffset = 0
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_getitem_nextcarry_64')
    funcPy(tocarry, mask, maskoffset, length, validwhen)
    outtocarry = [0, 1, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]

def test_awkward_ByteMaskedArray_getitem_nextcarry_outindex_64_1():
    tocarry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    outindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [1, 1, 1, 1, 1]
    maskoffset = 0
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_getitem_nextcarry_outindex_64')
    funcPy(tocarry, outindex, mask, maskoffset, length, validwhen)
    outtocarry = [0, 1, 2]
    for i in range(len(outtocarry)):
        assert tocarry[i] == outtocarry[i]
    outoutindex = [0.0, 1.0, 2.0]
    for i in range(len(outoutindex)):
        assert outindex[i] == outoutindex[i]

def test_awkward_ByteMaskedArray_toIndexedOptionArray64_1():
    toindex = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = [1, 1, 1, 1, 1]
    maskoffset = 0
    length = 3
    validwhen = True
    funcPy = getattr(tests.kernels, 'awkward_ByteMaskedArray_toIndexedOptionArray64')
    funcPy(toindex, mask, maskoffset, length, validwhen)
    outtoindex = [0, 1, 2]
    for i in range(len(outtoindex)):
        assert toindex[i] == outtoindex[i]

def test_awkward_Content_getitem_next_missing_jagged_getmaskstartstop_1():
    index_in = [1, 0, 0, 1, 1, 1, 0]
    index_in_offset = 1
    offsets_in = [14, 1, 27, 25, 3, 27, 7, 33, 18, 13]
    offsets_in_offset = 0
    mask_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_Content_getitem_next_missing_jagged_getmaskstartstop')
    funcPy(index_in, index_in_offset, offsets_in, offsets_in_offset, mask_out, starts_out, stops_out, length)
    outmask_out = [0, 1, 2]
    for i in range(len(outmask_out)):
        assert mask_out[i] == outmask_out[i]
    outstarts_out = [14, 1, 27]
    for i in range(len(outstarts_out)):
        assert starts_out[i] == outstarts_out[i]
    outstops_out = [1, 27, 25]
    for i in range(len(outstops_out)):
        assert stops_out[i] == outstops_out[i]

def test_awkward_MaskedArray64_getitem_next_jagged_project_1():
    index = [1, 0, 0, 1, 1, 1, 0]
    index_offset = 1
    starts_in = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts_offset = 0
    stops_in = [3, 1, 3, 2, 3]
    stops_offset = 0
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_MaskedArray64_getitem_next_jagged_project')
    funcPy(index, index_offset, starts_in, starts_offset, stops_in, stops_offset, starts_out, stops_out, length)
    outstarts_out = [2, 0, 2]
    for i in range(len(outstarts_out)):
        assert starts_out[i] == outstarts_out[i]
    outstops_out = [3, 1, 3]
    for i in range(len(outstops_out)):
        assert stops_out[i] == outstops_out[i]

def test_awkward_MaskedArray32_getitem_next_jagged_project_1():
    index = [1, 0, 0, 1, 1, 1, 0]
    index_offset = 1
    starts_in = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts_offset = 0
    stops_in = [3, 1, 3, 2, 3]
    stops_offset = 0
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_MaskedArray32_getitem_next_jagged_project')
    funcPy(index, index_offset, starts_in, starts_offset, stops_in, stops_offset, starts_out, stops_out, length)
    outstarts_out = [2, 0, 2]
    for i in range(len(outstarts_out)):
        assert starts_out[i] == outstarts_out[i]
    outstops_out = [3, 1, 3]
    for i in range(len(outstops_out)):
        assert stops_out[i] == outstops_out[i]

def test_awkward_MaskedArrayU32_getitem_next_jagged_project_1():
    index = [1, 0, 0, 1, 1, 1, 0]
    index_offset = 1
    starts_in = [2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
    starts_offset = 0
    stops_in = [3, 1, 3, 2, 3]
    stops_offset = 0
    starts_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    stops_out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length = 3
    funcPy = getattr(tests.kernels, 'awkward_MaskedArrayU32_getitem_next_jagged_project')
    funcPy(index, index_offset, starts_in, starts_offset, stops_in, stops_offset, starts_out, stops_out, length)
    outstarts_out = [2, 0, 2]
    for i in range(len(outstarts_out)):
        assert starts_out[i] == outstarts_out[i]
    outstops_out = [3, 1, 3]
    for i in range(len(outstops_out)):
        assert stops_out[i] == outstops_out[i]

