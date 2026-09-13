# ak.to_tensorflow

Defined in [awkward.operations.ak_to_tensorflow](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_tensorflow.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_tensorflow.py#L14).

#### ak.to_tensorflow(array)

Converts an Awkward Array into a TensorFlow Tensor, if possible.

If `array` contains any other data types (RecordArray for example) the
function raises a TypeError.

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  A TensorFlow Tensor with the same data as `array`, if the conversion is
  possible.
