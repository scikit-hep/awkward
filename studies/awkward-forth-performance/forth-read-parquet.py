import numpy as np

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

def read_rle(file_obj, header, bit_width, o):  # pragma: no cover
    count = header >> 1
    width = (bit_width + 7) // 8
    zero = np.zeros(4, dtype=np.int8)
    data = file_obj.read(width)
    zero[:len(data)] = data
    value = zero.view(np.int32)
    o.write_many(value, count)

def _mask_for_bits(i):  # pragma: no cover
    return (1 << i) - 1

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

    wrote = []

    print(f"{width=} {mask=}")

    while total >= width:
        print(f"{total=} {current_byte=} {bits_wnd_l=} {bits_wnd_r=} {data=}")
        # NOTE zero-padding could produce extra zero-values
        if bits_wnd_r >= 8:
            print("bits_wnd_r >= 8")
            bits_wnd_r -= 8
            bits_wnd_l -= 8
            data >>= 8
        elif bits_wnd_l - bits_wnd_r >= width:
            print("bits_wnd_l - bits_wnd_r >= width")
            wrote.append((data >> bits_wnd_r) & mask)
            print("WROTE BYTE ", (data >> bits_wnd_r) & mask)
            o.write_byte((data >> bits_wnd_r) & mask)
            total -= width
            bits_wnd_r += width
        elif current_byte + 1 < byte_count:
            print("current_byte + 1 < byte_count")
            current_byte += 1
            data |= (raw_bytes[current_byte] << bits_wnd_l)
            bits_wnd_l += 8

    raise Exception

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
    io_obj.loc = start + length
    return o.so_far()

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

def read_length(file_obj):  # pragma: no cover
    sub = file_obj.read(4)
    return sub[0] + sub[1]*256 + sub[2]*256*256 + sub[3]*256*256*256

def read_data(fobj, coding, count, bit_width):
    out = np.empty(count, dtype=np.int32)
    o = fastparquet.encoding.Numpy32(out)
    if coding == fastparquet.parquet_thrift.Encoding.RLE:
        while o.loc < count:
            read_rle_bit_packed_hybrid(fobj, bit_width, o=o)
    else:
        raise NotImplementedError('Encoding %s' % coding)
    return out

file = open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib1-split-jagged3.parquet", "rb")
parquetfile = fastparquet.ParquetFile(file)

for rg in parquetfile.filter_row_groups([]):
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
    # repetition_levels = fastparquet.core.read_rep(tmp, ph.data_page_header, parquetfile.schema, col.meta_data)
    repetition_levels = read_data(tmp,
                                  ph.data_page_header.repetition_level_encoding,
                                  ph.data_page_header.num_values,
                                  bit_width)[:ph.data_page_header.num_values]

    print(repetition_levels)
    print(values)
    break
