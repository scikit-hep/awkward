import time
import fastavro

begintime = time.time()

num = 0
next = 1

for x in fastavro.reader(open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged3.avro", "rb")):
    num += 1
    if 100.0 * num / 2097152.0 > next:
        print(next)
        next += 1

endtime = time.time()

print("fastavro zlib1-jagged3", endtime - begintime, "seconds")
