# ak.nan_to_none

Defined in [awkward.operations.ak_nan_to_none](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_nan_to_none.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_nan_to_none.py#L16).

#### ak.nan_to_none(array, \*, highlevel: [bool](https://docs.python.org/3/library/functions.html#bool) = True, behavior: awkward._typing.Mapping | [None](https://docs.python.org/3/library/constants.html#None) = None, attrs: awkward._typing.Mapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

Converts NaN (“not a number”) into None, i.e. missing values with option-type.

See also [`ak.nan_to_num`](sphinx-llm:b138b2f2922b4c2eaabf4955717204bf#ak.nan_to_num) to convert NaN or infinity to specified values.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with NaN (“not a number”) converted to None, i.e. missing
  values with option-type.
