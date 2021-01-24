import threading
import time
import sys

import uproot

num_workers = int(sys.argv[1])

print(f"prepare {num_workers}")

class Worker(threading.Thread):
    def __init__(self):
        super(Worker, self).__init__()
        self.objectarrays = []

    def add(self, branch, basket):
        self.objectarrays.append(uproot.interpretation.objects.ObjectArray(branch.interpretation.model, branch, {}, basket.byte_offsets, basket.data, 0))

    def run(self):
        for objectarray in self.objectarrays:
            for x in objectarray:
                pass

workers = [Worker() for i in range(num_workers)]

branch = uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged3.root:tree/branch")
baskets = [branch.basket(i) for i in range(72)]

for i, basket in enumerate(baskets):
    workers[i % num_workers].add(branch, basket)

print("begin")
begin = time.time()

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()

end = time.time()
print("end")

print(f"Uproot {num_workers} workers {end - begin} seconds")
