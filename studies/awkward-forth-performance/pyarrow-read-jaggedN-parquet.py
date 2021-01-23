import sys
import time

import awkward as ak

compress = sys.argv[1]
N = int(sys.argv[2])
is_split = sys.argv[3] == "split"

s = "-split" if is_split else ""
filename = f"/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/{compress}{s}-jagged{N}.parquet"

array = ak.from_parquet(filename, lazy=True)

begintime = time.time()
for partition in array.layout.partitions:
    tmp = partition.array

endtime = time.time()

print(f"pyarrow {compress}{s}-jagged{N}", endtime - begintime, "seconds")
