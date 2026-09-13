# ak.str.to_categorical

Defined in [awkward.operations.str.akstr_to_categorical](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_to_categorical.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/str/akstr_to_categorical.py#L12).

#### ak.str.to_categorical(array, \*, highlevel=True, behavior=None, attrs=None)

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Returns a dictionary-encoded version of the given array of strings.
Creates a categorical dataset, which has the following properties:

> * only distinct values (categories) are stored in their entirety,
> * pointers to those distinct values are represented by integers
>   (an [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) or [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray)
>   labeled with parameter `"__array__" = "categorical"`.

This is equivalent to R’s “factor”, and Pandas’s “categorical”.
It differs from generic uses of [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) and
[`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) in Awkward Arrays by the guarantee of no
duplicate categories and the `"categorical"` parameter.

Unlike Arrow’s `dictionary_encode`, this function has no `null_handling`
argument. This function’s behavior is like\`\`null_handling=”mask”\`\` (Arrow’s default).
It is not possible to encode null values in Awkward Array, as [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray)
cannot contain an option type node.

Note: this function does not raise an error if the `array` does not
contain any string or bytestring data.

Requires the pyarrow library and calls
[pyarrow.compute.dictionary_encode](https://arrow.apache.org/docs/python/generated/pyarrow.compute.dictionary_encode.html).
