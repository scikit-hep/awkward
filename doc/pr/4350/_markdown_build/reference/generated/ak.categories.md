# ak.categories

Defined in [awkward.operations.ak_categories](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_categories.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_categories.py#L12).

#### ak.categories(array, highlevel=True, \*, behavior=None, attrs=None)

Returns the categories of a categorical array.

If the `array` is categorical (contains [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) or
[`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) labeled with parameter
`"__array__" = "categorical"`), then this function returns its categories.

See also [`ak.is_categorical`](sphinx-llm:7fdfe526439549fab4b42580c8dc0f33#ak.is_categorical), [`ak.str.to_categorical`](sphinx-llm:ad92a02f22f4473ebb80165887f3828c#ak.str.to_categorical), [`ak.from_categorical`](sphinx-llm:8f5756af217b4853a3c68240d762ebf3#ak.from_categorical).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  The distinct category values of `array` (if it is categorical).
