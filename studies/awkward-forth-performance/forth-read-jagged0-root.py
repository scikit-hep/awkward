import time

import uproot
import awkward.forth
import awkward as ak
import numpy as np

branch = uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour1-jagged0.root:tree/branch")

begintime = time.time()

for basketid in range(branch.num_baskets):
    basket = branch.basket(basketid)
    start, stop = branch.basket_entry_start_stop(basketid)

    array2 = ak.Array(
        ak.layout.NumpyArray(basket.data.astype(np.float32))
    )

endtime = time.time()
print("AwkwardForth lzfour1-jagged0", stop, "entries", endtime - begintime, "seconds")
