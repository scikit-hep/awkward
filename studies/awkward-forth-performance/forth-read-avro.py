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

jagged0 = awkward.forth.ForthMachine32("""
input stream
output content float32

stream #f-> content
""")

with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged0.avro", "rb") as file:
    data = file.read()
    pos = data.index(0) + 1

    while pos + 16 < len(data):
        pos += 16

        pos, varint = decode_varint(pos, data)
        num_items = decode_zigzag(varint)

        pos, varint = decode_varint(pos, data)
        num_bytes = decode_zigzag(varint)

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        uncompressed = decompressor.decompress(data[pos : pos + num_bytes])
        uncompressed += decompressor.flush()
        pos += num_bytes

        jagged0.begin({"stream": uncompressed})
        jagged0.stack_push(num_items)
        jagged0.resume()
        assert jagged0.stack == []

        array = ak.Array(
            jagged0.output_NumpyArray("content"), check_valid=True
        )
        assert len(array) == num_items
        print(array)

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

with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged1.avro", "rb") as file:
    data = file.read()
    pos = data.index(0) + 1

    while pos + 16 < len(data):
        pos += 16

        pos, varint = decode_varint(pos, data)
        num_items = decode_zigzag(varint)

        pos, varint = decode_varint(pos, data)
        num_bytes = decode_zigzag(varint)

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        uncompressed = decompressor.decompress(data[pos : pos + num_bytes])
        uncompressed += decompressor.flush()
        pos += num_bytes

        jagged1.begin({"stream": uncompressed})
        jagged1.stack_push(num_items)
        jagged1.resume()
        assert jagged1.stack == []

        array = ak.Array(
            ak.layout.ListOffsetArray32(
                jagged1.output_Index32("offset0"),
                jagged1.output_NumpyArray("content")
            ), check_valid=True
        )
        assert len(array) == num_items
        print(array)

jagged2 = awkward.forth.ForthMachine32("""
input stream
output offset0 int32
output offset1 int32
output content float32

0 offset0 <- stack
0 offset1 <- stack

0 do
  stream zigzag-> stack
  dup offset0 +<- stack
  0 do
    stream zigzag-> stack
    dup offset1 +<- stack
    stream #f-> content
    stream b-> stack drop
  loop
  stream b-> stack drop
loop
""")

with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged2.avro", "rb") as file:
    data = file.read()
    pos = data.index(0) + 1

    while pos + 16 < len(data):
        pos += 16

        pos, varint = decode_varint(pos, data)
        num_items = decode_zigzag(varint)

        pos, varint = decode_varint(pos, data)
        num_bytes = decode_zigzag(varint)

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        uncompressed = decompressor.decompress(data[pos : pos + num_bytes])
        uncompressed += decompressor.flush()
        pos += num_bytes

        jagged2.begin({"stream": uncompressed})
        jagged2.stack_push(num_items)
        jagged2.resume()
        assert jagged2.stack == []

        array = ak.Array(
            ak.layout.ListOffsetArray32(
                jagged2.output_Index32("offset0"),
                ak.layout.ListOffsetArray32(
                    jagged2.output_Index32("offset1"),
                    jagged2.output_NumpyArray("content")
                )
            ), check_valid=True
        )
        assert len(array) == num_items
        print(array)

jagged3 = awkward.forth.ForthMachine32("""
input stream
output offset0 int32
output offset1 int32
output offset2 int32
output content float32

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
      stream #f-> content
      stream b-> stack drop
    loop
    stream b-> stack drop
  loop
  stream b-> stack drop
loop
""")

with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-jagged3.avro", "rb") as file:
    data = file.read()
    pos = data.index(0) + 1

    while pos + 16 < len(data):
        pos += 16

        pos, varint = decode_varint(pos, data)
        num_items = decode_zigzag(varint)

        pos, varint = decode_varint(pos, data)
        num_bytes = decode_zigzag(varint)

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        uncompressed = decompressor.decompress(data[pos : pos + num_bytes])
        uncompressed += decompressor.flush()
        pos += num_bytes

        jagged3.begin({"stream": uncompressed})
        jagged3.stack_push(num_items)
        jagged3.resume()
        assert jagged3.stack == []

        array = ak.Array(
            ak.layout.ListOffsetArray32(
                jagged3.output_Index32("offset0"),
                ak.layout.ListOffsetArray32(
                    jagged3.output_Index32("offset1"),
                    ak.layout.ListOffsetArray32(
                        jagged3.output_Index32("offset2"),
                        jagged3.output_NumpyArray("content")
                    )
                )
            ), check_valid=True
        )
        assert len(array) == num_items
        print(array)
