import time
import json

import numpy

import awkward
import awkward1

content = numpy.fromfile(open("data/sample-content.float32", "rb"), dtype=numpy.float32)
offsets1 = numpy.fromfile(open("data/sample-offsets1.int64", "rb"), dtype=numpy.int64)
offsets2 = numpy.fromfile(open("data/sample-offsets2.int64", "rb"), dtype=numpy.int64)
offsets3 = numpy.fromfile(open("data/sample-offsets3.int64", "rb"), dtype=numpy.int64)

array0 = awkward.JaggedArray.fromoffsets(offsets3,
             awkward.JaggedArray.fromoffsets(offsets2,
                 awkward.JaggedArray.fromoffsets(offsets1,
                     content)))

array1 = awkward1.layout.ListOffsetArray64(
             awkward1.layout.Index64(offsets3),
             awkward1.layout.ListOffsetArray64(
                 awkward1.layout.Index64(offsets2),
                 awkward1.layout.ListOffsetArray64(
                     awkward1.layout.Index64(offsets1),
                     awkward1.layout.NumpyArray(content))))

############################# slicing at each depth

if False:
    FRAC = 1
    REPS = 100
    starttime = time.time()
    for i in range(REPS):
        q = array0[1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array0[1:]\t", walltime, "sec;\t", len(content)/walltime/1e6, "million floats/sec")

    FRAC = 1
    REPS = 100
    starttime = time.time()
    for i in range(REPS):
        q = array1[1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1[1:]\t", walltime, "sec;\t", len(content)/walltime/1e6, "million floats/sec")

    FRAC = 1
    REPS = 10
    starttime = time.time()
    for i in range(REPS):
        q = array0[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array0[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e6, "million floats/sec")

    FRAC = 1
    REPS = 10
    starttime = time.time()
    for i in range(REPS):
        q = array1[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e6, "million floats/sec")

    print("array0[:, :, 1:] can't be done")

    FRAC = 1
    REPS = 5
    starttime = time.time()
    for i in range(REPS):
        q = array1[:, :, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1[:, :, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e6, "million floats/sec")

    print("array0[:, :, :, 1:] can't be done")

    REPS = 2
    FRAC = 2
    tmp = array1[:len(array1) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp[:, :, :, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1[:, :, :, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e6, "million floats/sec")

############################# slicing at first inner depth

if False:
    FRAC = 2
    REPS = 100
    tmp = array0.content.content[len(array0.content.content) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp  # array0.content.content[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array0.content.content[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e12, "trillion floats/sec")

    FRAC = 2
    REPS = 100
    tmp = array1.content.content[len(array1.content.content) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp  # array1.content.content[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1.content.content[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e12, "trillion floats/sec")

    FRAC = 2
    REPS = 100
    tmp = array0.content[len(array0.content) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp[:, 1:]   # array0.content[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array0.content[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e12, "trillion floats/sec")

    FRAC = 2
    REPS = 100
    tmp = array1.content[len(array1.content) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp[:, 1:]   # array1.content[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1.content[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e12, "trillion floats/sec")

    FRAC = 2
    REPS = 100
    tmp = array0[len(array0) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp[:, 1:]   # array0[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array0[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e12, "trillion floats/sec")

    FRAC = 2
    REPS = 100
    tmp = array1[len(array1) // FRAC]
    starttime = time.time()
    for i in range(REPS):
        q = tmp[:, 1:]   # array1[:, 1:]
    walltime = (time.time() - starttime)*FRAC/REPS
    print("array1[:, 1:]\t", walltime, "sec;\t", len(content)/walltime/1e12, "trillion floats/sec")

############################# from Python iterable
print("from Python iterable")

pyobj0 = awkward1.tolist(array1.content.content.content[:2000000])   # 200000000 takes 4 sec
sizepyobj0 = len(pyobj0)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(pyobj0)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(pyobj0)\t", walltime, "sec;\t", sizepyobj0/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromiter(pyobj0, initial=sizepyobj0+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromiter(pyobj0)\t", walltime, "sec;\t", sizepyobj0/walltime/1e6, "million floats/sec")

pyobj1 = awkward1.tolist(array1.content.content[:200000])   # 200000 takes 1 sec
sizepyobj1 = sum(len(x) for x in pyobj1)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(pyobj1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(pyobj1)\t", walltime, "sec;\t", sizepyobj1/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromiter(pyobj1, initial=sizepyobj1+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromiter(pyobj1)\t", walltime, "sec;\t", sizepyobj1/walltime/1e6, "million floats/sec")

pyobj2 = awkward1.tolist(array1.content[:200000])   # 20000 takes 1 sec
sizepyobj2 = sum(sum(len(y) for y in x) for x in pyobj2)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(pyobj2)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(pyobj2)\t", walltime, "sec;\t", sizepyobj2/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromiter(pyobj2, initial=sizepyobj2+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromiter(pyobj2)\t", walltime, "sec;\t", sizepyobj2/walltime/1e6, "million floats/sec")

pyobj3 = awkward1.tolist(array1[:20000])   # 2000 takes 1 sec
sizepyobj3 = sum(sum(sum(len(z) for z in y) for y in x) for x in pyobj3)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(pyobj3)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(pyobj3)\t", walltime, "sec;\t", sizepyobj3/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromiter(pyobj3, initial=sizepyobj3+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromiter(pyobj3)\t", walltime, "sec;\t", sizepyobj3/walltime/1e6, "million floats/sec")

############################# from JSON
print("from JSON")

pyobj0 = awkward1.tolist(array1.content.content.content[:2000000])   # 200000000 takes 4 sec
sizejobj0 = len(pyobj0)
jobj0 = json.dumps(pyobj0)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(json.loads(jobj0))
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(json.loads(jobj0))\t", walltime, "sec;\t", sizejobj0/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromjson(jobj0, initial=sizejobj0+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromjson(jobj0)\t", walltime, "sec;\t", sizejobj0/walltime/1e6, "million floats/sec")

pyobj1 = awkward1.tolist(array1.content.content[:200000])   # 200000 takes 1 sec
sizejobj1 = sum(len(x) for x in pyobj1)
jobj1 = json.dumps(pyobj1)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(json.loads(jobj1))
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(json.loads(jobj1))\t", walltime, "sec;\t", sizejobj1/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromjson(jobj1, initial=sizejobj1+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromjson(jobj1)\t", walltime, "sec;\t", sizejobj1/walltime/1e6, "million floats/sec")

pyobj2 = awkward1.tolist(array1.content[:200000])   # 20000 takes 1 sec
sizejobj2 = sum(sum(len(y) for y in x) for x in pyobj2)
jobj2 = json.dumps(pyobj2)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(json.loads(jobj2))
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(json.loads(jobj2))\t", walltime, "sec;\t", sizejobj2/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromjson(jobj2, initial=sizejobj2+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromjson(jobj2)\t", walltime, "sec;\t", sizejobj2/walltime/1e6, "million floats/sec")

pyobj3 = awkward1.tolist(array1[:20000])   # 2000 takes 1 sec
sizejobj3 = sum(sum(sum(len(z) for z in y) for y in x) for x in pyobj3)
jobj3 = json.dumps(pyobj3)

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward.fromiter(json.loads(jobj3))
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward.fromiter(json.loads(jobj3))\t", walltime, "sec;\t", sizejobj3/walltime/1e6, "million floats/sec")

FRAC = 1
REPS = 2
starttime = time.time()
for i in range(REPS):
    q = awkward1.fromjson(jobj3, initial=sizejobj3+1)
walltime = (time.time() - starttime)*FRAC/REPS
print("awkward1.fromjson(jobj3)\t", walltime, "sec;\t", sizejobj3/walltime/1e6, "million floats/sec")

