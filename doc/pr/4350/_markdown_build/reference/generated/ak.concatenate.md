# ak.concatenate

Defined in [awkward.operations.ak_concatenate](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_concatenate.py) on [line 35](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_concatenate.py#L35).

#### ak.concatenate(arrays, axis=0, \*, mergebool=True, highlevel=True, behavior=None, attrs=None)

Returns an array with the given arrays concatenated along an axis.

* **Parameters:**
  * **arrays** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied. The
    outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **mergebool** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, boolean and numeric data can be combined
    into the same buffer, losing information about False vs `0` and
    True vs `1`; otherwise, they are kept in separate buffers with
    distinct types (using an [`ak.contents.UnionArray`](sphinx-llm:dd41ab713e6a4626a50f392463f45ad1#ak.contents.UnionArray)).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with `arrays` concatenated. For `axis=0`, this means that
  one whole array follows another. For `axis=1`, it means that the `arrays`
  must have the same lengths and nested lists are each concatenated,
  element for element, and similarly for deeper levels.
