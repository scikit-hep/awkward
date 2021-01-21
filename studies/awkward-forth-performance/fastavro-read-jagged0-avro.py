import time
import fastavro

begintime = time.time()

num = 0
next = 1

for x in fastavro.reader(open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib0-jagged0.avro", "rb")):
    num += 1
    if 100.0 * num / 1073741824.0 > next:
        print(next)
        next += 1

endtime = time.time()

print("fastavro zlib0-jagged0", endtime - begintime, "seconds")
