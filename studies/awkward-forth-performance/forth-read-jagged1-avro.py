import zlib
import struct
import time

import numpy as np
import awkward as ak
import awkward.forth

data = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged1.avro", np.uint8)

def decompress(data, pos):
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    uncompressed = decompressor.decompress(data[pos : pos + num_bytes])
    uncompressed += decompressor.flush()
    return uncompressed

def decode_varint(pos, data):
    shift = 0
    result = 0
    while True:
        i = data[pos]
        pos += 1
        result |= (i & 0x7f) << shift
        shift += 7
        if not (i & 0x80):
            break
    return pos, result

def decode_zigzag(n):
    return (n >> 1) ^ (-(n & 1))

jagged1 = awkward.forth.ForthMachine32("""
input stream
output offset0 int32
output content float32

0 offset0 <- stack

0 do
  stream zigzag-> stack
  dup offset0 +<- stack
  stream #f-> content
  stream b-> stack drop
loop
""")

begintime = time.time()

pos = 0
while data[pos] != 0:
    pos += 1
pos += 1

while pos + 16 < len(data):
    pos += 16

    pos, varint = decode_varint(pos, data)
    num_items = decode_zigzag(varint)

    pos, varint = decode_varint(pos, data)
    num_bytes = decode_zigzag(varint)

    uncompressed = decompress(data, pos)

    pos += num_bytes

    jagged1.begin({"stream": uncompressed})
    jagged1.stack_push(num_items)
    jagged1.resume()

    array = ak.Array(
        ak.layout.ListOffsetArray32(
            jagged1.output_Index32("offset0"),
            jagged1.output_NumpyArray("content")
        )
    )

endtime = time.time()

print("AwkwardForth zlib1-jagged1", endtime - begintime, "seconds")
