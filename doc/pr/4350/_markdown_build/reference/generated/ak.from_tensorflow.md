# ak.from_tensorflow

Defined in [awkward.operations.ak_from_tensorflow](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_tensorflow.py) on [line 13](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_tensorflow.py#L13).

#### ak.from_tensorflow(array)

Converts a TensorFlow Tensor into an Awkward Array.

A tensor on a GPU is not copied: its buffer is shared through DLPack. A
tensor on a CPU is copied, because a NumPy array is mutable and a
TensorFlow tensor is not.

If `array` contains any other data types the function raises an error.

* **Parameters:**
  **array** – (TensorFlow Tensor):
  Tensor to convert into an Awkward Array.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) built from the given TensorFlow Tensor.
