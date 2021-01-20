import numpy as np
import awkward as ak

content = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", np.float32)
offsets1 = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", np.int64)
offsets2 = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets2.int64", np.int64)

# jagged2

events_per_basket = 219309

partitions = []
start2 = stop2 = 0
while start2 < len(offsets2) - 1:
    stop2 = int(min(stop2 + events_per_basket, len(offsets2) - 1))

    o2 = offsets2[start2 : stop2 + 1]
    o1 = offsets1[o2[0] : o2[-1] + 1]
    c = content[o1[0] : o1[-1]]

    o2 = (o2 - o2[0]).astype(np.int32)
    o1 = (o1 - o1[0]).astype(np.int32)

    partitions.append(ak.Array(
        ak.layout.ListOffsetArray32(
            ak.layout.Index32(o2),
            ak.layout.ListOffsetArray32(
                ak.layout.Index32(o1),
                ak.layout.NumpyArray(c)
            )
        ), check_valid=True
    ))

    start2 = stop2

for level in [None]:  # [9, 1]:
    print("level", level)
    ak.to_parquet(
        ak.partitioned(partitions),
        "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour" + str(level) + "-jagged2.parquet",
        list_to32=True,
        compression="LZ4",
        compression_level=level,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=100*1024**2,
    )
    print("level", level, "split")
    ak.to_parquet(
        ak.partitioned(partitions),
        "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour" + str(level) + "-split-jagged2.parquet",
        list_to32=True,
        compression="LZ4",
        compression_level=level,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=100*1024**2,
        use_byte_stream_split=True,
    )

# for level in [0]:
#     print("level", level)
#     ak.to_parquet(
#         ak.partitioned(partitions),
#         "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-jagged2.parquet",
#         list_to32=True,
#         compression="NONE",
#         compression_level=None,
#         use_dictionary=False,
#         write_statistics=False,
#         data_page_size=100*1024**2,
#     )
#     print("level", level, "split")
#     ak.to_parquet(
#         ak.partitioned(partitions),
#         "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-split-jagged2.parquet",
#         list_to32=True,
#         compression="NONE",
#         compression_level=None,
#         use_dictionary=False,
#         write_statistics=False,
#         data_page_size=100*1024**2,
#         use_byte_stream_split=True,
#     )
