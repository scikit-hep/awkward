# ak.from_raggedtensor

Defined in [awkward.operations.ak_from_raggedtensor](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_raggedtensor.py) on [line 13](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_raggedtensor.py#L13).

#### ak.from_raggedtensor(array)

Converts a TensorFlow RaggedTensor into an Awkward Array.

The underlying buffers (flat values and row splits) are shared through
DLPack if the RaggedTensor is on a GPU; on a CPU, they are copied.

If `array` contains any other data types the function raises an error.

* **Parameters:**
  **array** – (`tensorflow.RaggedTensor`):
  RaggedTensor to convert into an  Awkward Array.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given TensorFlow RaggedTensor.
