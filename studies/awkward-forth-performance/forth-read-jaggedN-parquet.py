import time
import sys
import subprocess

import numba as nb
import numpy as np
import awkward as ak
import awkward.forth

import fastparquet
import fastparquet.core
import fastparquet.thrift_structures

compress = sys.argv[1]
N = int(sys.argv[2])
is_split = sys.argv[3] == "split"

s = "-split" if is_split else ""
filename = f"/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/{compress}{s}-jagged{N}.parquet"

subprocess.call(f"vmtouch -t {filename} > /dev/null", shell=True)
subprocess.call(f"vmtouch {filename} | fgrep Pages", shell=True)

file = open(filename, "rb")
parquetfile = fastparquet.ParquetFile(file)

if is_split:
    def unsplit(data):
        values = np.empty(len(data), np.uint8)
        quarter = len(data) // 4
        values[::4]  = data[:quarter]
        values[1::4] = data[quarter:2*quarter]
        values[2::4] = data[2*quarter:3*quarter]
        values[3::4] = data[3*quarter:]
        return values.view(np.float32)
else:
    def unsplit(data):
        return data.view(np.float32)

def bit_width_of(value):
    for i in range(0, 64):
        if value == 0:
            return i
        value >>= 1

bit_width = bit_width_of(N)

if N > 0:
    read_repetition_levels = awkward.forth.ForthMachine32("""
input stream
output reps uint8

stream I-> stack
begin
  stream varint-> stack
  dup 1 and 0= if
    ( run-length encoding )
    stream {rle_format}-> reps
    1 rshift 1-
    reps dup
  else
    ( bit-packed )
    1 rshift 8 *
    stream #{bit_width}bit-> reps
  then
  dup stream pos 4 - <=
until
""".format(bit_width=bit_width, rle_byte_width=1, rle_format="B"))

jagged1 = awkward.forth.ForthMachine32("""
input reps
output offsets0 int32
variable count0

begin
  reps b-> stack

  dup 1 = if
    1 count0 +!
  then
  0 = if
    count0 @ offsets0 +<- stack
    1 count0 !
  then

  reps end
until

count0 @ offsets0 +<- stack
""")

jagged2 = awkward.forth.ForthMachine32("""
input reps
output offsets0 int32
output offsets1 int32
variable count0
variable count1

begin
  reps b-> stack

  dup 2 = if
    1 count1 +!
  then
  dup 1 = if
    count1 @ offsets1 +<- stack
    1 count1 !
    1 count0 +!
  then
  0 = if
    count1 @ offsets1 +<- stack
    1 count1 !
    count0 @ offsets0 +<- stack
    1 count0 !
  then

  reps end
until

count1 @ offsets1 +<- stack
count0 @ offsets0 +<- stack
""")

jagged3 = awkward.forth.ForthMachine32("""
input reps
output offsets0 int32 output offsets1 int32 output offsets2 int32
variable count0 variable count1 variable count2

begin
  reps b-> stack

  dup 3 = if
    1 count2 +!
  then
  dup 2 = if
    count2 @ offsets2 +<- stack
    1 count2 !
    1 count1 +!
  then
  dup 1 = if
    count2 @ offsets2 +<- stack
    1 count2 !
    count1 @ offsets1 +<- stack
    1 count1 !
    1 count0 +!
  then
  0 = if
    count2 @ offsets2 +<- stack
    1 count2 !
    count1 @ offsets1 +<- stack
    1 count1 !
    count0 @ offsets0 +<- stack
    1 count0 !
  then

  reps end
until

count2 @ offsets2 +<- stack
count1 @ offsets1 +<- stack
count0 @ offsets0 +<- stack
""")

begintime = time.time()
for rg in parquetfile.filter_row_groups([]):
    col = rg.columns[0]
    file.seek(col.meta_data.data_page_offset)
    ph = fastparquet.thrift_structures.read_thrift(file, fastparquet.thrift_structures.parquet_thrift.PageHeader)

    raw_bytes = np.frombuffer(fastparquet.core._read_page(file, ph, col.meta_data), np.uint8)

    data = raw_bytes[-(ph.data_page_header.num_values * 4):]
    values = unsplit(data)

    max_repetition_level = col.meta_data.path_in_schema.count("list")
    assert max_repetition_level == N
    bit_width = bit_width_of(max_repetition_level)
    rep_length, = raw_bytes[:4].view(np.uint32)

    if N != 0:
        read_repetition_levels.run({"stream": raw_bytes})
        repetition_levels = np.asarray(read_repetition_levels["reps"])[:len(values)]
        
    if N == 0:
        array = ak.layout.NumpyArray(values)
    elif N == 1:
        jagged1.run({"reps": repetition_levels})
        array = ak.layout.ListOffsetArray32(
            ak.layout.Index32(jagged1["offsets0"]),
            ak.layout.NumpyArray(values)
        )
    elif N == 2:
        jagged2.run({"reps": repetition_levels})
        array = ak.layout.ListOffsetArray32(
            ak.layout.Index32(jagged2["offsets0"]),
            ak.layout.ListOffsetArray32(
                ak.layout.Index32(jagged2["offsets1"]),
                ak.layout.NumpyArray(values)
            )
        )
    elif N == 3:
        jagged3.run({"reps": repetition_levels})
        array = ak.layout.ListOffsetArray32(
            ak.layout.Index32(jagged3["offsets0"]),
            ak.layout.ListOffsetArray32(
                ak.layout.Index32(jagged3["offsets1"]),
                ak.layout.ListOffsetArray32(
                    ak.layout.Index32(jagged3["offsets2"]),
                    ak.layout.NumpyArray(values)
                )
            )
        )
    tmp = ak.Array(array)

endtime = time.time()

print(f"AwkwardForth {compress}{s}-jagged{N}", endtime - begintime, "seconds")
