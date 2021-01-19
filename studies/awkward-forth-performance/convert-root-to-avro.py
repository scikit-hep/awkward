import uproot
import fastavro
import gc


schema = fastavro.parse_schema({
    "name": "jagged3",
    "namespace": "org.awkward-array",
    "type": "array",
    "items": {"type": "array", "items": {"type": "array", "items": "double"}}
})

def generate():
    for arrays in uproot.iterate(
        "sample-jagged3.root:jagged3",
        "branch",
        step_size=2*8812,
        library="np",
    ):
        for x in arrays["branch"]:
            yield [[list(z) for z in y] for y in x]

with open("sample-jagged3.avro", "wb") as out:
    fastavro.writer(
        out,
        schema,
        generate(),
        codec="deflate",
        codec_compression_level=1,
        sync_interval=37374063,
    )

gc.collect()

schema = fastavro.parse_schema({
    "name": "jagged2",
    "namespace": "org.awkward-array",
    "type": "array",
    "items": {"type": "array", "items": "double"}
})

def generate():
    for arrays in uproot.iterate(
        "sample-jagged2.root:jagged2",
        "branch",
        step_size=2*51151,
        library="np",
    ):
        for x in arrays["branch"]:
            yield [list(y) for y in x]

with open("sample-jagged2.avro", "wb") as out:
    fastavro.writer(
        out,
        schema,
        generate(),
        codec="deflate",
        codec_compression_level=1,
        sync_interval=27109203,
    )

gc.collect()

# schema = fastavro.parse_schema({
#     "name": "jagged1",
#     "namespace": "org.awkward-array",
#     "type": "array",
#     "items": "double"
# })

# def generate():
#     for arrays in uproot.iterate(
#         "sample-jagged1.root:jagged1",
#         "branch",
#         step_size=2*415535,
#         library="np",
#     ):
#         for x in arrays["branch"]:
#             yield list(x)

# with open("sample-jagged1.avro", "wb") as out:
#     fastavro.writer(
#         out,
#         schema,
#         generate(),
#         codec="deflate",
#         codec_compression_level=1,
#         sync_interval=27425337,
#     )

# gc.collect()

# schema = fastavro.parse_schema({
#     "name": "jagged0",
#     "namespace": "org.awkward-array",
#     "type": "double"
# })

# def generate():
#     for arrays in uproot.iterate(
#         "sample-jagged0.root:jagged0",
#         "branch",
#         step_size=2*3918766,
#         library="np",
#     ):
#         for x in arrays["branch"]:
#             yield x

# with open("sample-jagged0.avro", "wb") as out:
#     fastavro.writer(
#         out,
#         schema,
#         generate(),
#         codec="deflate",
#         codec_compression_level=1,
#         sync_interval=31350128,
#     )
