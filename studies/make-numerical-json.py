import numpy

maxbytes = 4*1024**3   # 4 GB

content = numpy.empty(maxbytes // numpy.dtype(numpy.float32).itemsize, dtype=numpy.float32)
for i in range(0, len(content), len(content) // 8):
    content[i : i + len(content) // 8] = numpy.random.normal(0, 1, len(content) // 8)
content.tofile(open("sample-content.float32", "wb"))

offsets1 = numpy.arange(0, len(content) + 8, 8, dtype=numpy.int64)
offsets1[1:-1] += numpy.random.randint(0, 8, len(offsets1) - 2)
offsets1.tofile(open("sample-offsets1.int64", "wb"))

offsets2 = numpy.arange(0, len(offsets1) - 1 + 8, 8, dtype=numpy.int64)
offsets2[1:-1] += numpy.random.randint(0, 8, len(offsets2) - 2)
offsets2.tofile(open("sample-offsets2.int64", "wb"))

offsets3 = numpy.arange(0, len(offsets2) - 1 + 8, 8, dtype=numpy.int64)
offsets3[1:-1] += numpy.random.randint(0, 8, len(offsets3) - 2)
offsets3.tofile(open("sample-offsets3.int64", "wb"))

# import awkward
# a = awkward.JaggedArray.fromoffsets(offsets3, awkward.JaggedArray.fromoffsets(offsets2, awkward.JaggedArray.fromoffsets(offsets1, content)))
