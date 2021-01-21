import time

import numba as nb
import numpy as np
import awkward.forth

import fastparquet
import fastparquet.core
import fastparquet.encoding
import fastparquet.thrift_structures

file = open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-split-jagged3.parquet", "rb")
parquetfile = fastparquet.ParquetFile(file)

def bit_width_of(value):
    for i in range(0, 64):
        if value == 0:
            return i
        value >>= 1

bit_width = bit_width_of(3)

read_repetition_levels = awkward.forth.ForthMachine32("""
input stream
output replevels uint8

stream I-> stack
begin
  stream varint-> stack
  dup 1 and 0= if
    ( run-length encoding )
    stream {rle_format}-> replevels
    1 rshift 1-
    replevels dup
  else
    ( bit-packed )
    1 rshift 8 *
    stream #{bit_width}bit-> replevels
  then
  dup stream pos 4 - <=
until
""".format(bit_width=bit_width, rle_byte_width=1, rle_format="B"))

cumulative_theirs = 0.0
cumulative_ours = 0.0

for i, rg in enumerate(parquetfile.filter_row_groups([])):
    col = rg.columns[0]
    file.seek(col.meta_data.data_page_offset)
    ph = fastparquet.thrift_structures.read_thrift(file, fastparquet.thrift_structures.parquet_thrift.PageHeader)

    raw_bytes = np.frombuffer(fastparquet.core._read_page(file, ph, col.meta_data), np.uint8)

    data = raw_bytes[-(ph.data_page_header.num_values * 4):]
    values = np.empty(len(data), np.uint8)
    quarter = len(data) // 4
    values[::4]  = data[:quarter]
    values[1::4] = data[quarter:2*quarter]
    values[2::4] = data[2*quarter:3*quarter]
    values[3::4] = data[3*quarter:]
    values = values.view(np.float32)

    max_repetition_level = col.meta_data.path_in_schema.count("list")
    bit_width = bit_width_of(max_repetition_level)
    rep_length, = raw_bytes[:4].view(np.uint32)

    begintime_theirs = time.time()
    repetition_levels = fastparquet.core.read_rep(
        fastparquet.encoding.Numpy8(raw_bytes), ph.data_page_header, parquetfile.schema, col.meta_data
    )
    endtime_theirs = time.time()
    if i > 5:
        cumulative_theirs += endtime_theirs - begintime_theirs

    rle_byte_width = (bit_width + 7) // 8
    assert rle_byte_width == 1

    begintime_ours = time.time()
    read_repetition_levels.run({"stream": raw_bytes})
    my_repetition_levels = np.asarray(read_repetition_levels["replevels"])[:len(values)]
    endtime_ours = time.time()
    if i > 5:
        cumulative_ours += endtime_ours - begintime_ours

    if i > 5:
        print(cumulative_theirs, cumulative_ours, cumulative_ours / cumulative_theirs)
    print(values)
    print(repetition_levels)
    print(my_repetition_levels)
    assert np.array_equal(repetition_levels, my_repetition_levels)
