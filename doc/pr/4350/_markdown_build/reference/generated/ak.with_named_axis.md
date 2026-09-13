# ak.with_named_axis

Defined in [awkward.operations.ak_with_named_axis](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_named_axis.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_named_axis.py#L20).

#### ak.with_named_axis(array, named_axis: awkward._namedaxis.AxisTuple | awkward._namedaxis.AxisMapping, \*, highlevel=True, behavior=None, attrs=None)

Returns an array with named axes attached.

This function does not change the array in-place. If the new name is None,
then the array is returned as it is.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **named_axis** – AxisTuple | AxisMapping: Names to give to the array axis; this assigns
    the `"__named_axis__"` attr. If None, any existing name is unset.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) (or low-level equivalent, if
  `highlevel=False`) with named axes attached.
