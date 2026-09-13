# ak.cartesian

Defined in [awkward.operations.ak_cartesian](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_cartesian.py) on [line 30](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_cartesian.py#L30).

#### ak.cartesian(arrays, axis=1, \*, nested=None, parameters=None, with_name=None, highlevel=True, behavior=None, attrs=None)

Computes the Cartesian product (cross product) of data from a set of arrays.

This operation creates records (if `arrays` is a dict) or tuples (if
`arrays` is another kind of iterable) that hold the combinations of
elements, and it can introduce new levels of nesting.

The order of the output is fixed: it is always lexicographical in the
order that the `arrays` are written.

To emulate an SQL or Pandas “group by” operation, put the keys that you
wish to group by *first* and use `nested=[0]` or `nested=[n]` to group by
unique n-tuples. If necessary, record keys can later be reordered with a
list of strings in [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__).

To get list index positions in the tuples/records, rather than data from
the original `arrays`, use [`ak.argcartesian`](sphinx-llm:1876594663754700b6a556da188e003d#ak.argcartesian) instead of [`ak.cartesian`](sphinx-llm:aaa706eae6ec430ca41e0be16250ad14). The
[`ak.argcartesian`](sphinx-llm:1876594663754700b6a556da188e003d#ak.argcartesian) form can be particularly useful as nested indexing in
[`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__).

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
    items for a chosen set of keys from the `array` dict or integer
    slots of the `array` iterable.
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
  An array holding the Cartesian product (i.e. cross product) of data
  from a set of `arrays`.

### Examples

As a simple example with `axis=0`, the Cartesian product of

```pycon
>>> one = ak.Array([1, 2, 3])
>>> two = ak.Array(["a", "b"])
```

is

```pycon
>>> ak.cartesian([one, two], axis=0).show()
[(1, 'a'),
 (1, 'b'),
 (2, 'a'),
 (2, 'b'),
 (3, 'a'),
 (3, 'b')]
```

With nesting, a new level of nested lists is created to group combinations
that share the same element from `one` into the same list.

```pycon
>>> ak.cartesian([one, two], axis=0, nested=True).show()
[[(1, 'a'), (1, 'b')],
 [(2, 'a'), (2, 'b')],
 [(3, 'a'), (3, 'b')]]
```

The primary purpose of this function, however, is to compute a different
Cartesian product for each element of an array: in other words, `axis=1`.
The following arrays each have four elements.

```pycon
>>> one = ak.Array([[1, 2, 3], [], [4, 5], [6]])
>>> two = ak.Array([["a", "b"], ["c"], ["d"], ["e", "f"]])
```

The default `axis=1` produces 6 pairs from the Cartesian product of
`[1, 2, 3]` and `["a", "b"]`, 0 pairs from `[]` and `["c"]`, 1 pair from
`[4, 5]` and `["d"]`, and 1 pair from `[6]` and `["e", "f"]`.

```pycon
>>> ak.cartesian([one, two]).show()
[[(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')],
 [],
 [(4, 'd'), (5, 'd')],
 [(6, 'e'), (6, 'f')]]
```

The nesting depth is the same as the original arrays; with `nested=True`,
the nesting depth is increased by 1 and tuples are grouped by their
first element.

```pycon
>>> ak.cartesian([one, two], nested=True).show()
[[[(1, 'a'), (1, 'b')], [(2, 'a'), (2, ...)], [(3, 'a'), (3, 'b')]],
 [],
 [[(4, 'd')], [(5, 'd')]],
 [[(6, 'e'), (6, 'f')]]]
```

These tuples are [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) nodes with unnamed fields. To
name the fields, we can pass `one` and `two` in a dict, rather than a list.

```pycon
>>> ak.cartesian({"x": one, "y": two}).show()
[[{x: 1, y: 'a'}, {x: 1, y: 'b'}, {...}, ..., {x: 3, y: 'a'}, {x: 3, y: 'b'}],
 [],
 [{x: 4, y: 'd'}, {x: 5, y: 'd'}],
 [{x: 6, y: 'e'}, {x: 6, y: 'f'}]]
```

With more than two elements in the Cartesian product, `nested` can specify
which are grouped and which are not. For example,

```pycon
>>> one = ak.Array([1, 2, 3, 4])
>>> two = ak.Array([1.1, 2.2, 3.3])
>>> three = ak.Array(["a", "b"])
```

can be left entirely ungrouped:

```pycon
>>> ak.cartesian([one, two, three], axis=0).show()
[(1, 1.1, 'a'),
 (1, 1.1, 'b'),
 (1, 2.2, 'a'),
 (1, 2.2, 'b'),
 (1, 3.3, 'a'),
 (1, 3.3, 'b'),
 (2, 1.1, 'a'),
 (2, 1.1, 'b'),
 (2, 2.2, 'a'),
 (2, 2.2, 'b'),
 ...,
 (3, 2.2, 'b'),
 (3, 3.3, 'a'),
 (3, 3.3, 'b'),
 (4, 1.1, 'a'),
 (4, 1.1, 'b'),
 (4, 2.2, 'a'),
 (4, 2.2, 'b'),
 (4, 3.3, 'a'),
 (4, 3.3, 'b')]
```

can be grouped by `one` (adding 1 more dimension):

```pycon
>>> ak.cartesian([one, two, three], axis=0, nested=[0]).show()
[[(1, 1.1, 'a'), (1, 1.1, 'b'), (1, 2.2, 'a')],
 [(1, 2.2, 'b'), (1, 3.3, 'a'), (1, 3.3, 'b')],
 [(2, 1.1, 'a'), (2, 1.1, 'b'), (2, 2.2, 'a')],
 [(2, 2.2, 'b'), (2, 3.3, 'a'), (2, 3.3, 'b')],
 [(3, 1.1, 'a'), (3, 1.1, 'b'), (3, 2.2, 'a')],
 [(3, 2.2, 'b'), (3, 3.3, 'a'), (3, 3.3, 'b')],
 [(4, 1.1, 'a'), (4, 1.1, 'b'), (4, 2.2, 'a')],
 [(4, 2.2, 'b'), (4, 3.3, 'a'), (4, 3.3, 'b')]]
```

can be grouped by `one` and `two` (adding 2 more dimensions):

```pycon
>>> ak.cartesian([one, two, three], axis=0, nested=[0, 1]).show()
[[[(1, 1.1, 'a'), (1, 1.1, 'b')], [...], [(1, 3.3, 'a'), (1, 3.3, ...)]],
 [[(2, 1.1, 'a'), (2, 1.1, 'b')], [...], [(2, 3.3, 'a'), (2, 3.3, ...)]],
 [[(3, 1.1, 'a'), (3, 1.1, 'b')], [...], [(3, 3.3, 'a'), (3, 3.3, ...)]],
 [[(4, 1.1, 'a'), (4, 1.1, 'b')], [...], [(4, 3.3, 'a'), (4, 3.3, ...)]]]
```

or grouped by unique `one`-`two` pairs (adding 1 more dimension):

```pycon
>>> ak.cartesian([one, two, three], axis=0, nested=[1]).show()
[[(1, 1.1, 'a'), (1, 1.1, 'b')],
 [(1, 2.2, 'a'), (1, 2.2, 'b')],
 [(1, 3.3, 'a'), (1, 3.3, 'b')],
 [(2, 1.1, 'a'), (2, 1.1, 'b')],
 [(2, 2.2, 'a'), (2, 2.2, 'b')],
 [(2, 3.3, 'a'), (2, 3.3, 'b')],
 [(3, 1.1, 'a'), (3, 1.1, 'b')],
 [(3, 2.2, 'a'), (3, 2.2, 'b')],
 [(3, 3.3, 'a'), (3, 3.3, 'b')],
 [(4, 1.1, 'a'), (4, 1.1, 'b')],
 [(4, 2.2, 'a'), (4, 2.2, 'b')],
 [(4, 3.3, 'a'), (4, 3.3, 'b')]]
```
