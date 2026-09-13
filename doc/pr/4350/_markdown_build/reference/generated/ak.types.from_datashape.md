# ak.types.from_datashape

Defined in [awkward.types.type](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/type.py) on [line 312](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/types/type.py#L312).

#### ak.types.from_datashape(datashape, highlevel=True)

Parses `datashape` (str) and returns a [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type) object, the inverse of
calling `str` on a [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type).

If `highlevel=True`, and the type string starts with a number (e.g. ‘1000 \* …’),
the return type is [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType), representing an `ak.highlevel.Array`.

If `highlevel=True` and the type string starts with a record indicator (e.g. `{`),
the return type is [`ak.types.ScalarType`](sphinx-llm:ad298acb59fd4c1bb1bf103171505226#ak.types.ScalarType) with an [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) content,
representing a scalar `ak.highlevel.Record` rather than an array of them.

Other strings (e.g. starting with `var *`, `?`, `option`, etc.) are not compatible
with `highlevel=True`; an exception would be raised.

If `highlevel=False`, the type is assumed to represent a layout (e.g. a number
indicates a [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType), rather than a [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType)).
