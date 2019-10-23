import os
os.chdir("..")
print(os.getcwd())

import numpy
import awkward1

content = numpy.fromfile(open("studies/sample-content.float32", "rb"), dtype=numpy.float32)
offsets1 = numpy.fromfile(open("studies/sample-offsets1.int64", "rb"), dtype=numpy.int64)
offsets2 = numpy.fromfile(open("studies/sample-offsets2.int64", "rb"), dtype=numpy.int64)
offsets3 = numpy.fromfile(open("studies/sample-offsets3.int64", "rb"), dtype=numpy.int64)

array = awkward1.layout.ListOffsetArray64(
            awkward1.layout.Index64(offsets3),
            awkward1.layout.ListOffsetArray64(
                awkward1.layout.Index64(offsets2),
                awkward1.layout.ListOffsetArray64(
                    awkward1.layout.Index64(offsets1),
                    awkward1.layout.NumpyArray(content))))

array.tojson("studies/sample-jagged3.json", maxdecimals=5)
array.content.tojson("studies/sample-jagged2.json", maxdecimals=5)
array.content.content.tojson("studies/sample-jagged1.json", maxdecimals=5)
array.content.content.content.tojson("studies/sample-jagged0.json", maxdecimals=5)
