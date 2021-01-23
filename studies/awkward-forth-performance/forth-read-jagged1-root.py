import time

import uproot
import awkward.forth
import awkward as ak
import numpy as np

jagged1 = awkward.forth.ForthMachine32("""
input data
input byte_offsets
output offsets0 int32
output content float32

0 offsets0 <- stack

begin
  byte_offsets i-> stack
  6 + data seek
  data !i-> stack
  dup offsets0 +<- stack
  data #!f-> content
again
""")

branch = uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour1-jagged1.root:tree/branch")

begintime = time.time()

for basketid in range(branch.num_baskets):
    basket = branch.basket(basketid)
    start, stop = branch.basket_entry_start_stop(basketid)

    jagged1.run(
        {"data": np.copy(basket.data), "byte_offsets": basket.byte_offsets},
        raise_read_beyond=False,
        raise_seek_beyond=False,
    )
    array2 = ak.Array(
        ak.layout.ListOffsetArray32(
            jagged1.output_Index32("offsets0"),
            jagged1.output_NumpyArray("content")
        )
    )

endtime = time.time()
print("AwkwardForth lzfour1-jagged1", stop, "entries", endtime - begintime, "seconds")
