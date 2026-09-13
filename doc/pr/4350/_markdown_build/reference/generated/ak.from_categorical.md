# ak.from_categorical

Defined in [awkward.operations.ak_from_categorical](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_categorical.py) on [line 12](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_from_categorical.py#L12).

#### ak.from_categorical(array, \*, highlevel=True, behavior=None, attrs=None)

Replaces categorical data with equivalent non-categorical data.

This is a metadata-only operation; the running time does not scale with the
size of the dataset. (Conversion to categorical is expensive; conversion
from categorical is cheap.)

See also [`ak.is_categorical`](sphinx-llm:7fdfe526439549fab4b42580c8dc0f33#ak.is_categorical), [`ak.categories`](sphinx-llm:f80f94bfb360496785af4b13b6b2cec8#ak.categories), [`ak.str.to_categorical`](sphinx-llm:ad92a02f22f4473ebb80165887f3828c#ak.str.to_categorical).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with categorical data replaced by the equivalent non-categorical
  data (by removing the label that declares it as such).
