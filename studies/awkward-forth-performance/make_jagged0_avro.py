import numpy as np
import awkward as ak
import uproot
import fastavro

content = np.memmap("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/sample-content.float32", np.float32)

array = content

events_per_basket = 16777197

schema = fastavro.parse_schema({
    "name": "jagged0",
    "namespace": "org.awkward-array",
    "type": "float"
})

for level in 9, 1, 0:
    with open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib" + str(level) + "-jagged0.avro", "wb") as out:
        fastavro.writer(
            out,
            schema,
            array,
            codec="deflate",
            codec_compression_level=level,
            sync_interval=67108788,
        )
