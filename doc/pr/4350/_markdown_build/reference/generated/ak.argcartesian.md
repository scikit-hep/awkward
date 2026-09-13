# ak.argcartesian

Defined in [awkward.operations.ak_argcartesian](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argcartesian.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argcartesian.py#L16).

#### ak.argcartesian(arrays, axis=1, \*, nested=None, parameters=None, with_name=None, highlevel=True, behavior=None, attrs=None)

Computes the Cartesian product of arrays, returning integer indexes.

All of the parameters for [`ak.cartesian`](sphinx-llm:e515482efbae4eb28bb9f06e625d21c4#ak.cartesian) apply equally to [`ak.argcartesian`](sphinx-llm:6dd43f5962b44d8bba57904a665bd63c),
so see the [`ak.cartesian`](sphinx-llm:e515482efbae4eb28bb9f06e625d21c4#ak.cartesian) documentation for a more complete description.

* **Parameters:**
  * **arrays** (*mapping* *or* *sequence* *of* *arrays*) – Each value in this mapping or
    sequence can be any array-like data that [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes.
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied. The
    outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **nested** (*None* *,* *True* *,* *False* *, or* *iterable* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – If None or
    False, all combinations of elements from the `arrays` are
    produced at the same level of nesting; if True, they are grouped
    in nested lists by combinations that share a common item from
    each of the `arrays`; if an iterable of str or int, group common
    items for a chosen set of keys from the `array` dict or slots
    of the `array` iterable.
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
  An array of integer indexes into the Cartesian product (i.e. cross product)
  of a set of `arrays`, like [`ak.cartesian`](sphinx-llm:e515482efbae4eb28bb9f06e625d21c4#ak.cartesian) but for use with
  [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__).

### Examples

For example, the Cartesian product of

```pycon
>>> one = ak.Array([1.1, 2.2, 3.3])
>>> two = ak.Array(["a", "b"])
```

is

```pycon
>>> ak.cartesian([one, two], axis=0).show()
[(1.1, 'a'),
 (1.1, 'b'),
 (2.2, 'a'),
 (2.2, 'b'),
 (3.3, 'a'),
 (3.3, 'b')]
```

But with argcartesian, only the indexes are returned.

```pycon
>>> ak.argcartesian([one, two], axis=0).show()
[(0, 0),
 (0, 1),
 (1, 0),
 (1, 1),
 (2, 0),
 (2, 1)]
```

These are the indexes that can select the items that go into the actual
Cartesian product.

```pycon
>>> one_index, two_index = ak.unzip(ak.argcartesian([one, two], axis=0))
>>> one[one_index]
<Array [1.1, 1.1, 2.2, 2.2, 3.3, 3.3] type='6 * float64'>
>>> two[two_index]
<Array ['a', 'b', 'a', 'b', 'a', 'b'] type='6 * string'>
```
