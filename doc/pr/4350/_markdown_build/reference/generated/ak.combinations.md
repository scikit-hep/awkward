# ak.combinations

Defined in [awkward.operations.ak_combinations](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_combinations.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_combinations.py#L20).

#### ak.combinations(array, n, \*, replacement=False, axis=1, fields=None, parameters=None, with_name=None, highlevel=True, behavior=None, attrs=None)

Computes combinations of `n` items from an array, without replacement.

If the normal Cartesian product is thought of as an `n` dimensional tensor,
these represent the “upper triangle” of sets without repetition. If
`replacement=True`, the diagonal of this “upper triangle” is included.

To get list index positions in the tuples/records, rather than data from
the original `array`, use [`ak.argcombinations`](sphinx-llm:ecff05623c824acf8dd10f3cb43c5322#ak.argcombinations) instead of [`ak.combinations`](sphinx-llm:298f698616d1407292e834db7f88491d).
The [`ak.argcombinations`](sphinx-llm:ecff05623c824acf8dd10f3cb43c5322#ak.argcombinations) form can be particularly useful as nested indexing
in [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **n** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The number of items to choose in each list: `2` chooses
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
  An array holding the Cartesian product (i.e. cross product) of `array`
  with itself, restricted to combinations sampled without replacement.

### Examples

As a simple example with `axis=0`, consider the following

```pycon
>>> array = ak.Array(["a", "b", "c", "d", "e"])
```

The combinations choose `2` are:

```pycon
>>> ak.combinations(array, 2, axis=0).show()
[('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
             ('b', 'c'), ('b', 'd'), ('b', 'e'),
                         ('c', 'd'), ('c', 'e'),
                                     ('d', 'e')]
```

Including the diagonal allows pairs like `('a', 'a')`.

```pycon
>>> ak.combinations(array, 2, axis=0, replacement=True).show()
[('a', 'a'), ('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
             ('b', 'b'), ('b', 'c'), ('b', 'd'), ('b', 'e'),
                         ('c', 'c'), ('c', 'd'), ('c', 'e'),
                                     ('d', 'd'), ('d', 'e'),
                                                 ('e', 'e')]
```

The combinations choose `3` can’t be easily arranged as a triangle
in two dimensions.

```pycon
>>> ak.combinations(array, 3, axis=0).show()
[('a', 'b', 'c'),
 ('a', 'b', 'd'),
 ('a', 'b', 'e'),
 ('a', 'c', 'd'),
 ('a', 'c', 'e'),
 ('a', 'd', 'e'),
 ('b', 'c', 'd'),
 ('b', 'c', 'e'),
 ('b', 'd', 'e'),
 ('c', 'd', 'e')]
```

Including the (three-dimensional) diagonal allows triples like
`('a', 'a', 'a')`, but also `('a', 'a', 'b')`, `('a', 'b', 'b')`, etc.,
but not `('a', 'b', 'a')`. All combinations are in the same order as
the original array.

```pycon
>>> ak.combinations(array, 3, axis=0, replacement=True).show()
[('a', 'a', 'a'),
 ('a', 'a', 'b'),
 ('a', 'a', 'c'),
 ('a', 'a', 'd'),
 ('a', 'a', 'e'),
 ('a', 'b', 'b'),
 ('a', 'b', 'c'),
 ('a', 'b', 'd'),
 ('a', 'b', 'e'),
 ('a', 'c', 'c'),
 ...,
 ('c', 'c', 'd'),
 ('c', 'c', 'e'),
 ('c', 'd', 'd'),
 ('c', 'd', 'e'),
 ('c', 'e', 'e'),
 ('d', 'd', 'd'),
 ('d', 'd', 'e'),
 ('d', 'e', 'e'),
 ('e', 'e', 'e')]
```

The primary purpose of this function, however, is to compute a different
set of combinations for each element of an array: in other words, `axis=1`.
The following has a different number of items in each element.

```pycon
>>> array = ak.Array([[1, 2, 3, 4], [], [5], [6, 7, 8]])
```

There are 6 ways to choose pairs from 4 elements, 0 ways to choose pairs
from 0 elements, 0 ways to choose pairs from 1 element, and 3 ways to
choose pairs from 3 elements.

```pycon
>>> ak.combinations(array, 2).show()
[[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
 [],
 [],
 [(6, 7), (6, 8), (7, 8)]]
```

Note, however, that the combinatorics isn’t determined by equality of
the data themselves, but by their placement in the array. For example,
even if all elements of an array are equal, the output has the same
structure.

```pycon
>>> same = ak.Array([[7, 7, 7, 7], [], [7], [7, 7, 7]])
>>> ak.combinations(same, 2).show()
[[(7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7)],
 [],
 [],
 [(7, 7), (7, 7), (7, 7)]]
```

To get records instead of tuples, pass a set of field names to `fields`.

```pycon
>>> ak.combinations(array, 2, fields=["x", "y"]).show()
[
 [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 4},
                    {'x': 2, 'y': 3}, {'x': 2, 'y': 4},
                                      {'x': 3, 'y': 4}],
 [],
 [],
 [{'x': 6, 'y': 7}, {'x': 6, 'y': 8},
                    {'x': 7, 'y': 8}]]
```

This operation can be constructed from [`ak.argcartesian`](sphinx-llm:1876594663754700b6a556da188e003d#ak.argcartesian) and other
primitives:

```pycon
>>> left, right = ak.unzip(ak.argcartesian([array, array]))
>>> keep = left < right
>>> result = ak.zip([array[left][keep], array[right][keep]])
>>> result.show()
[
 [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
 [],
 [],
 [(6, 7), (6, 8), (7, 8)]]
```

but it is frequently needed for data analysis, and the logic of which
indexes to `keep` (above) gets increasingly complicated for large `n`.
