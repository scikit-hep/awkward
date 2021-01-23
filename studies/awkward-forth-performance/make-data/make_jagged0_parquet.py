import numpy as np
import awkward as ak

content = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", np.float32)

# jagged0

events_per_basket = 16777197

partitions = []
start0 = stop0 = 0
while start0 < len(content):
    stop0 = int(min(stop0 + events_per_basket, len(content)))

    c = content[start0 : stop0]

    partitions.append(ak.Array(
        ak.layout.NumpyArray(c), check_valid=True
    ))

    start0 = stop0

for level in [None]:  # [9, 1]:
    print("level", level)
    ak.to_parquet(
        ak.partitioned(partitions),
        "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour" + str(level) + "-jagged0.parquet",
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
        "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour" + str(level) + "-split-jagged0.parquet",
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
#         "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-jagged0.parquet",
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
#         "/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-split-jagged0.parquet",
#         list_to32=True,
#         compression="NONE",
#         compression_level=None,
#         use_dictionary=False,
#         write_statistics=False,
#         data_page_size=100*1024**2,
#         use_byte_stream_split=True,
#     )
