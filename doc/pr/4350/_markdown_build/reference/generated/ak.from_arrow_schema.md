# ak.from_arrow_schema

Defined in [awkward.operations.ak_from_arrow_schema](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_arrow_schema.py) on [line 13](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_arrow_schema.py#L13).

#### ak.from_arrow_schema(schema)

Converts an Apache Arrow schema into an Awkward Form.

Because awkward uses numpy’s dtype system, timestamp types do not have
timezones. If encountering timestamp types with timezones in the input
arrow data, they will be silently dropped.

See also [`ak.to_arrow`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow), [`ak.to_arrow_table`](sphinx-llm:b370f32af9534c74b92e6e448b182ab2#ak.to_arrow_table), [`ak.from_arrow`](sphinx-llm:a01617c5c7474de09fdc0ddb61b2cced#ak.from_arrow), [`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet), [`ak.from_parquet`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet).

* **Parameters:**
  **schema** (`pyarrow.Schema`) – Apache Arrow schema to convert into an Awkward Form.
* **Returns:**
  An [`ak.forms.Form`](sphinx-llm:6295086f75dd4713a4cbcfec3594d1a1#ak.forms.Form) built from the given Apache Arrow schema.
