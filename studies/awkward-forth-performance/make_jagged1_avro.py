import numpy as np
import awkward as ak
import uproot
import fastavro
import time
import json

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
    start = stop = 0
    while start < len(array):
        stop = min(stop + events_per_basket, len(array))
        chunk = json.loads(ak.to_json(array[start:stop]))
        for x in chunk:
            yield x
        print(int(round(100 * stop / len(array))), "percent",
              time.asctime(time.localtime()))
        start = stop

for level in [9, 1]:  # 9, 1, 0:
    print("level", level)
    with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/lzfour" + str(level) + "-jagged1.avro", "wb") as out:
        fastavro.writer(
            out,
            schema,
            generate(),
            codec="lz4",  # "deflate",
            codec_compression_level=level,
            sync_interval=45633959,
        )
