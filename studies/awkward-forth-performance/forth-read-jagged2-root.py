import time

import uproot
import awkward.forth
import awkward as ak
import numpy as np

jagged2 = awkward.forth.ForthMachine32("""
input data
input byte_offsets
output offsets1 int32
output offsets0 int32
output content float32

0 offsets1 <- stack
0 offsets0 <- stack

begin
  byte_offsets i-> stack
  6 + data seek
  data !i-> stack
  dup offsets1 +<- stack
  0 do
    data !i-> stack
    dup offsets0 +<- stack
    data #!f-> content
  loop
again
""")

branch = uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged2.root:tree/branch")

begintime = time.time()

for basketid in range(branch.num_baskets):
    basket = branch.basket(basketid)
    start, stop = branch.basket_entry_start_stop(basketid)

    jagged2.run(
        {"data": basket.data, "byte_offsets": basket.byte_offsets},
        raise_read_beyond=False,
        raise_seek_beyond=False,
    )
    array2 = ak.Array(
        ak.layout.ListOffsetArray32(
            jagged2.output_Index32("offsets1"),
            ak.layout.ListOffsetArray32(
                jagged2.output_Index32("offsets0"),
                jagged2.output_NumpyArray("content")
            )
        )
    )

endtime = time.time()
print("AwkwardForth zlib0-jagged2", stop, "entries", endtime - begintime, "seconds")
