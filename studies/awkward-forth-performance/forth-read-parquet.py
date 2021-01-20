import numba as nb
import numpy as np
import awkward.forth

import fastparquet
import fastparquet.core
import fastparquet.encoding
import fastparquet.thrift_structures

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

def bit_width_of(value):
    for i in range(0, 64):
        if value == 0:
            return i
        value >>= 1

class Whatever(Exception): pass

@nb.njit
def read_rle(file_obj, header, bit_width, o):  # pragma: no cover
    count = header >> 1
    width = (bit_width + 7) // 8
    zero = np.zeros(4, dtype=np.int8)
    data = file_obj.read(width)
    zero[:len(data)] = data
    value = zero.view(np.int32)
    print(count)
    print(width)
    print(value)
    print("o.loc", o.loc)
    o.write_many(value, count)
    raise Whatever("read_rle")

# fastparquet.encoding.read_rle = read_rle

@nb.njit
def _mask_for_bits(i):  # pragma: no cover
    return (1 << i) - 1

@nb.njit
def read_bitpacked(file_obj, header, width, o):  # pragma: no cover
    num_groups = header >> 1
    count = num_groups * 8
    byte_count = (width * count) // 8
    raw_bytes = file_obj.read(byte_count)
    mask = _mask_for_bits(width)
    current_byte = 0
    data = raw_bytes[current_byte]
    bits_wnd_l = 8
    bits_wnd_r = 0
    total = byte_count * 8
    # output = []
    while total >= width:
        # NOTE zero-padding could produce extra zero-values
        if bits_wnd_r >= 8:
            bits_wnd_r -= 8
            bits_wnd_l -= 8
            data >>= 8
        elif bits_wnd_l - bits_wnd_r >= width:
            # output.append((data >> bits_wnd_r) & mask)
            o.write_byte((data >> bits_wnd_r) & mask)
            total -= width
            bits_wnd_r += width
        elif current_byte + 1 < byte_count:
            current_byte += 1
            data |= (raw_bytes[current_byte] << bits_wnd_l)
            bits_wnd_l += 8

@nb.njit
def read_rle_bit_packed_hybrid(io_obj, width, length=None, o=None):  # pragma: no cover
    if length is None:
        length = read_length(io_obj)
    start = io_obj.loc
    while io_obj.loc - start < length and o.loc < o.len:
        header = read_unsigned_var_int(io_obj)
        if header & 1 == 0:
            read_rle(io_obj, header, width, o)
        else:
            read_bitpacked(io_obj, header, width, o)
        print(io_obj.loc - start, length, start)
    io_obj.loc = start + length
    return o.so_far()

# fastparquet.encoding.read_rle_bit_packed_hybrid = read_rle_bit_packed_hybrid

@nb.njit
def read_unsigned_var_int(file_obj):  # pragma: no cover
    result = 0
    shift = 0
    while True:
        byte = file_obj.read_byte()
        result |= ((byte & 0x7F) << shift)
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result

@nb.njit
def read_length(file_obj):  # pragma: no cover
    sub = file_obj.read(4)
    return sub[0] + sub[1]*256 + sub[2]*256*256 + sub[3]*256*256*256

def read_data(fobj, coding, count, bit_width):
    out = np.full(count, -1, dtype=np.int32)
    o = fastparquet.encoding.Numpy32(out)
    print("o.loc", o.loc)
    try:
        if coding == fastparquet.parquet_thrift.Encoding.RLE:
            while o.loc < count:
                read_rle_bit_packed_hybrid(fobj, bit_width, o=o)
        else:
            raise NotImplementedError('Encoding %s' % coding)
    except Whatever:
        pass
    return out

# fastparquet.core.read_data = read_data

file = open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-split-jagged3.parquet", "rb")
parquetfile = fastparquet.ParquetFile(file)

for i, rg in enumerate(parquetfile.filter_row_groups([])):
    print("rg")
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

    tmp = fastparquet.encoding.Numpy8(raw_bytes)
    repetition_levels = None
    repetition_levels = fastparquet.core.read_rep(tmp, ph.data_page_header, parquetfile.schema, col.meta_data)
    # repetition_levels = read_data(tmp, ph.data_page_header.repetition_level_encoding, ph.data_page_header.num_values, bit_width)[:ph.data_page_header.num_values]
    print(repetition_levels)
    print(values)

    rle_byte_width = (bit_width + 7) // 8
    if rle_byte_width == 1:
        rle_format = "B"
    elif rle_byte_width == 2:
        rle_format = "H"
    elif rle_byte_width == 4:
        rle_format = "I"
    elif rle_byte_width == 8:
        rle_format = "Q"
    else:
        raise Exception("RLE can't read {0} byte integers", rle_byte_width)

    jagged3 = awkward.forth.ForthMachine32(f"""
    input stream
    output replevels uint8

    stream I-> stack
    begin
      stream varint-> stack
      dup 1 and 0= if
        ( run-length encoding )
        1 rshift
        0 do
          stream {rle_format}-> replevels
          {rle_byte_width} negate stream skip
        loop
        {rle_byte_width} stream skip
      else
        ( bit-packed )
        1 rshift 8 *
        stream #{bit_width}bit-> replevels
      then
      dup stream pos 4 - <=
    until
    """.format(bit_width=bit_width, rle_byte_width=rle_byte_width, rle_format=rle_format))
    jagged3.run({"stream": raw_bytes})
    my_repetition_levels = np.asarray(jagged3["replevels"])[:len(values)]

    print(repetition_levels)
    print(my_repetition_levels)
    assert np.array_equal(repetition_levels, my_repetition_levels)
