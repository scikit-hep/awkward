import zlib
import struct

import awkward as ak
import awkward.forth


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


deserializer = awkward.forth.ForthMachine32("""
input stream
output offset0 int32
output offset1 int32
output offset2 int32
output content float64

0 offset0 <- stack
0 offset1 <- stack
0 offset2 <- stack

0 do
  stream zigzag-> stack
  dup offset0 +<- stack
  0 do
    stream zigzag-> stack
    dup offset1 +<- stack
    0 do
      stream zigzag-> stack
      dup offset2 +<- stack
      stream #d-> content
      stream b-> stack drop
    loop
    stream b-> stack drop
  loop
  stream b-> stack drop
loop
""")

with open("/home/jpivarski/storage/data/chep-2019-jagged-jagged-jagged/sample-jagged3.avro", "rb") as file:
    data = file.read()
    pos = data.index(0) + 1

    while pos + 16 < len(data):
        pos += 16

        pos, varint = decode_varint(pos, data)
        num_items = decode_zigzag(varint)
        print(num_items)

        pos, varint = decode_varint(pos, data)
        num_bytes = decode_zigzag(varint)

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        uncompressed = decompressor.decompress(data[pos : pos + num_bytes])
        uncompressed += decompressor.flush()
        pos += num_bytes

        deserializer.begin({"stream": uncompressed})
        deserializer.stack_push(num_items)
        deserializer.resume()
        assert deserializer.stack == []

        array = ak.Array(
            ak.layout.ListOffsetArray32(
                deserializer.output_Index32("offset0"),
                ak.layout.ListOffsetArray32(
                    deserializer.output_Index32("offset1"),
                    ak.layout.ListOffsetArray32(
                        deserializer.output_Index32("offset2"),
                        deserializer.output_NumpyArray("content")
                    )
                )
            ), check_valid=True
        )
        print(array)
