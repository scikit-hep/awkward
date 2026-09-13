# ak.isclose

Defined in [awkward.operations.ak_isclose](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_isclose.py) on [line 18](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_isclose.py#L18).

#### ak.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False, \*, highlevel=True, behavior=None, attrs=None)

Returns a boolean array of element-wise approximate-equality between two arrays.

Implements [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
for Awkward Arrays.

* **Parameters:**
  * **a** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **b** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **rtol** ([*float*](https://docs.python.org/3/library/functions.html#float)) – The relative tolerance parameter.
  * **atol** ([*float*](https://docs.python.org/3/library/functions.html#float)) – The absolute tolerance parameter.
  * **equal_nan** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Whether to compare `NaN` as equal. If True, `NaN` in
    `a` will be considered equal to `NaN` in `b`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array of booleans, True where `a` and `b` are approximately equal
  within the given tolerances.
