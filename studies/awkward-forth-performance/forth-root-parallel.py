import threading
import time
import sys

import uproot
import awkward.forth
import awkward as ak
import numpy as np

num_workers = int(sys.argv[1])

print(f"prepare {num_workers}")

class Worker(threading.Thread):
    def __init__(self):
        super(Worker, self).__init__()
        self.baskets = []
        self.jagged3 = awkward.forth.ForthMachine32("""
input data
input byte_offsets
output offsets2 int32
output offsets1 int32
output offsets0 int32
output content float32

0 offsets2 <- stack
0 offsets1 <- stack
0 offsets0 <- stack

begin
  byte_offsets i-> stack
  6 + data seek
  data !i-> stack
  dup offsets2 +<- stack
  0 do
    data !i-> stack
    dup offsets1 +<- stack
    0 do
      data !i-> stack
      dup offsets0 +<- stack
      data #!f-> content
    loop
  loop
again
""")

    def add(self, basket):
        self.baskets.append(basket)

    def run(self):
        for basket in self.baskets:
            self.jagged3.run(
                {"data": basket.data, "byte_offsets": basket.byte_offsets},
                raise_read_beyond=False,
                raise_seek_beyond=False,
                raise_skip_beyond=False,
            )

workers = [Worker() for i in range(num_workers)]

branch = uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged3.root:tree/branch")
baskets = [branch.basket(i) for i in range(72)]

for i, basket in enumerate(baskets):
    workers[i % num_workers].add(basket)

print("begin")
begin = time.time()

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()

end = time.time()
print("end")

print(f"Forth {num_workers} workers {end - begin} seconds")
