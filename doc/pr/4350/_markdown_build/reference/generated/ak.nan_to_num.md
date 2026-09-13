# ak.nan_to_num

Defined in [awkward.operations.ak_nan_to_num](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_nan_to_num.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_nan_to_num.py#L19).

#### ak.nan_to_num(array, copy: [bool](https://docs.python.org/3/library/functions.html#bool) = True, nan=0.0, posinf=None, neginf=None, \*, highlevel: [bool](https://docs.python.org/3/library/functions.html#bool) = True, behavior: awkward._typing.Mapping | [None](https://docs.python.org/3/library/constants.html#None) = None, attrs: awkward._typing.Mapping | [None](https://docs.python.org/3/library/constants.html#None) = None)

Replaces NaN and infinite values with finite numbers in floating-point arrays.

See also [`ak.nan_to_none`](sphinx-llm:9ecf4aa6eb61432b9955ca347e71f9ff#ak.nan_to_none) to convert NaN to None, i.e. missing values with
option-type.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Ignored (Awkward Arrays are immutable).
  * **nan** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* [*float*](https://docs.python.org/3/library/functions.html#float) *,* *broadcastable array*) – Value to be used to fill `NaN` values.
  * **posinf** (*None* *,* [*int*](https://docs.python.org/3/library/functions.html#int) *,* [*float*](https://docs.python.org/3/library/functions.html#float) *,* *broadcastable array*) – Value to be used to fill positive infinity
    values. If None, positive infinities are replaced with a very large number.
  * **neginf** (*None* *,* [*int*](https://docs.python.org/3/library/functions.html#int) *,* [*float*](https://docs.python.org/3/library/functions.html#float) *,* *broadcastable array*) – Value to be used to fill negative infinity
    values. If None, negative infinities are replaced with a very small number.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with NaN (“not a number”) or infinity replaced by the specified
  finite values, following
  [np.nan_to_num](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
  for Awkward Arrays.
