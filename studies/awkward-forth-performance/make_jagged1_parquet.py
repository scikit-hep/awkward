import numpy as np
import awkward as ak

content = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", np.float32)
offsets1 = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", np.int64)

# jagged1

events_per_basket = 1342176

partitions = []
start1 = stop1 = 0
while start1 < len(offsets1) - 1:
    stop1 = int(min(stop1 + events_per_basket, len(offsets1) - 1))

    o1 = offsets1[start1 : stop1 + 1]
    c = content[o1[0] : o1[-1]]

    o1 = (o1 - o1[0]).astype(np.int32)

    partitions.append(ak.Array(
        ak.layout.ListOffsetArray32(
            ak.layout.Index32(o1),
            ak.layout.NumpyArray(c)
        ), check_valid=True
    ))

    start1 = stop1

# for level in [9, 1]:
#     print("level", level)
#     ak.to_parquet(
#         ak.partitioned(partitions),
#         "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-jagged1.parquet",
#         list_to32=True,
#         compression="GZIP",
#         compression_level=level,
#         use_dictionary=False,
#         write_statistics=False,
#         data_page_size=100*1024**2,
#     )
#     print("level", level, "split")
#     ak.to_parquet(
#         ak.partitioned(partitions),
#         "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-split-jagged1.parquet",
#         list_to32=True,
#         compression="GZIP",
#         compression_level=level,
#         use_dictionary=False,
#         write_statistics=False,
#         data_page_size=100*1024**2,
#         use_byte_stream_split=True,
#     )

for level in [0]:
    print("level", level)
    ak.to_parquet(
        ak.partitioned(partitions),
        "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-jagged1.parquet",
        list_to32=True,
        compression="NONE",
        compression_level=None,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=100*1024**2,
    )
    print("level", level, "split")
    ak.to_parquet(
        ak.partitioned(partitions),
        "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-split-jagged1.parquet",
        list_to32=True,
        compression="NONE",
        compression_level=None,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=100*1024**2,
        use_byte_stream_split=True,
    )
