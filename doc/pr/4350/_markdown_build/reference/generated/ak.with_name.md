# ak.with_name

Defined in [awkward.operations.ak_with_name](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_name.py) on [line 15](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_name.py#L15).

#### ak.with_name(array, name, \*, highlevel=True, behavior=None, attrs=None)

Returns an array or record with the `__record__` parameter set to the given name.

This function does not change the array in-place. If the new name is None,
then an array without a name is returned.

The records or tuples may be nested within multiple levels of nested lists.
If records are nested within records, only the outermost are affected.

Setting the `"__record__"` parameter makes it possible to add behaviors
to the data; see [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) and [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for a more complete
description.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *None*) – Name to give to the records or tuples; this assigns
    the `"__record__"` parameter. If None, any existing name is unset.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) or [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) (or low-level equivalent, if
  `highlevel=False`) with a new name.
