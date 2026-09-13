# ak.is_categorical

Defined in [awkward.operations.ak_is_categorical](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_is_categorical.py) on [line 11](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_is_categorical.py#L11).

#### ak.is_categorical(array)

Returns True if the array is categorical.

If the `array` is categorical (contains [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) or
[`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) labeled with parameter
`"__array__" = "categorical"`), then this function returns True; otherwise,
it returns False.

See also [`ak.categories`](sphinx-llm:f80f94bfb360496785af4b13b6b2cec8#ak.categories), [`ak.str.to_categorical`](sphinx-llm:ad92a02f22f4473ebb80165887f3828c#ak.str.to_categorical), [`ak.from_categorical`](sphinx-llm:8f5756af217b4853a3c68240d762ebf3#ak.from_categorical).

* **Parameters:**
  **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
* **Returns:**
  True if `array` is categorical, False otherwise.
