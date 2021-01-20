import numpy as np
import awkward as ak
import uproot
import fastavro

content = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", np.float32)
offsets1 = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-offsets1.int64", np.int64)

array = ak.Array(
    ak.layout.ListOffsetArray64(
        ak.layout.Index64(offsets1),
        ak.layout.NumpyArray(content)
    ), check_valid=True
)

events_per_basket = 1342176

schema = fastavro.parse_schema({
    "name": "jagged1",
    "namespace": "org.awkward-array",
    "type": "array",
    "items": "float"
})

def generate():
    next = 10
    for i, x in enumerate(array):
        yield list(x)
        if 100 * i / len(array) >= next:
            print(round(100 * i / len(array)), "percent")
            next += 10

for level in 9, 1, 0:
    print("level", level)
    with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-jagged1.avro", "wb") as out:
        fastavro.writer(
            out,
            schema,
            generate(),
            codec="deflate",
            codec_compression_level=level,
            sync_interval=45633959,
        )
