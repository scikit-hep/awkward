# ak.to_raggedtensor

Defined in [awkward.operations.ak_to_raggedtensor](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_raggedtensor.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_raggedtensor.py#L14).

#### ak.to_raggedtensor(array)

Converts an Awkward Array into a TensorFlow RaggedTensor, if possible.

If `array` contains any other data types (RecordArray for example) the
function raises an error.

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  A TensorFlow RaggedTensor with the same data as `array`, if the conversion is
  possible.
