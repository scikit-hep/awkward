# ak.zeros_like

Defined in [awkward.operations.ak_zeros_like](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_zeros_like.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_zeros_like.py#L17).

#### ak.zeros_like(array, \*, dtype=None, including_unknown=False, highlevel=True, behavior=None, attrs=None)

Returns an array with the same structure as the input, filled with zeros.

This is the equivalent of NumPy’s `np.zeros_like` for Awkward Arrays.

(There is no equivalent of NumPy’s `np.empty_like` because Awkward Arrays
are immutable.)

See [`ak.full_like`](sphinx-llm:53b0700609594eabb4653a809a676a50#ak.full_like) for details, and see also [`ak.ones_like`](sphinx-llm:a6efac4195034ee78a35a71b042de1dd#ak.ones_like).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **dtype** (*None* *or* *NumPy dtype*) – Overrides the data type of the result.
  * **including_unknown** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, the `unknown` type is considered
    a value type and is converted to a zero-length array of the
    specified dtype; if False, `unknown` will remain `unknown`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with the same structure as `array`, with every value replaced
  by zero.
