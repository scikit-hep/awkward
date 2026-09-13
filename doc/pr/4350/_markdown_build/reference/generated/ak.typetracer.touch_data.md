# ak.typetracer.touch_data

Defined in [awkward.typetracer](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/typetracer.py) on [line 138](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/typetracer.py#L138).

#### ak.typetracer.touch_data(array: awkward._typing.Any, \*, highlevel: [bool](https://docs.python.org/3/library/functions.html#bool) = True, behavior: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [None](https://docs.python.org/3/library/constants.html#None) = None, attrs: [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), awkward._typing.Any] | [None](https://docs.python.org/3/library/constants.html#None) = None) → awkward.highlevel.Array | awkward.highlevel.Record

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.

Recursively touches the data and returns a shallow copy of the given array.
