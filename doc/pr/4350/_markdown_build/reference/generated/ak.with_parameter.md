# ak.with_parameter

Defined in [awkward.operations.ak_with_parameter](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_parameter.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_with_parameter.py#L14).

#### ak.with_parameter(array, parameter, value, \*, highlevel=True, behavior=None, attrs=None)

Returns an array with the given parameter set on the outermost layout node.

Note that a “new array” is a lightweight shallow copy, not a duplication
of large data buffers.

You can also remove a single parameter with this function, since setting
a parameter to None is equivalent to removing it.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **parameter** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Name of the parameter to set on that array.
  * **value** (*JSON*) – Value of the parameter to set on that array.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with a parameter set on the outermost
  node of its [`ak.Array.layout`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.layout).
