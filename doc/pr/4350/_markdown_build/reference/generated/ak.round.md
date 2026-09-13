# ak.round

Defined in [awkward.operations.ak_round](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_round.py) on [line 18](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_round.py#L18).

#### ak.round(array, decimals: [int](https://docs.python.org/3/library/functions.html#int) = 0, out=UNSUPPORTED, highlevel=True, behavior=None, attrs=None)

Rounds each array element to the given number of decimals.

Implements [np.round](https://numpy.org/doc/stable/reference/generated/numpy.round.html)
for Awkward Arrays.

* **Parameters:**
  * **array** – array_like
    Input array.
  * **decimals** – int, optional
    Number of decimal places to round to (default: 0).  If
    decimals is negative, it specifies the number of positions to
    the left of the decimal point.
  * **out** – unsupported optional argument
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with each element rounded to the given number of `decimals`.
