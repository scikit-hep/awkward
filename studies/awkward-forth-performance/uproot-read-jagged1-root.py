import time

import uproot

branch = uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged1.root:tree/branch")

begintime = time.time()

for basketid in range(branch.num_baskets):
    if basketid == 20: break

    basket = branch.basket(basketid)
    start, stop = branch.basket_entry_start_stop(basketid)

    array1 = branch.array(entry_start=start, entry_stop=stop, library="np")

endtime = time.time()
print("Uproot zlib1-jagged1", stop, "entries", endtime - begintime, "seconds")
