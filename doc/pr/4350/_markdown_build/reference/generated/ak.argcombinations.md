# ak.argcombinations

Defined in [awkward.operations.ak_argcombinations](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argcombinations.py) on [line 17](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argcombinations.py#L17).

#### ak.argcombinations(array, n, \*, replacement=False, axis=1, fields=None, parameters=None, with_name=None, highlevel=True, behavior=None, attrs=None)

Computes combinations of `n` items from an array, returning integer indexes.

The motivation and uses of this function are similar to those of
[`ak.argcartesian`](sphinx-llm:1876594663754700b6a556da188e003d#ak.argcartesian). See [`ak.combinations`](sphinx-llm:e2edae471dd3475f90dd9ae86f4e03a3#ak.combinations) and [`ak.argcartesian`](sphinx-llm:1876594663754700b6a556da188e003d#ak.argcartesian) for a more
complete description.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **n** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The number of items to choose from each list: `2` chooses
    unique pairs, `3` chooses unique triples, etc.
  * **replacement** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, combinations that include the same
    item more than once are allowed; otherwise each item in a
    combinations is strictly unique.
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied. The
    outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **fields** (*None* *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, the pairs/triples/etc. are
    tuples with unnamed fields; otherwise, these `fields` name the
    fields. The number of `fields` must be equal to `n`.
  * **parameters** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Parameters for the new
    [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node that is created by this operation.
  * **with_name** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Assigns a `"__record__"` name to the new
    [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node that is created by this operation
    (overriding `parameters`, if necessary).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array of integer indexes into the combinations of `array` with
  itself (sampled without replacement), like [`ak.combinations`](sphinx-llm:e2edae471dd3475f90dd9ae86f4e03a3#ak.combinations) but for use
  with [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__).
