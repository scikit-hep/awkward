# ak.count

Defined in [awkward.operations.ak_count](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_count.py) on [line 22](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_count.py#L22).

#### ak.count(array, axis=None, \*, keepdims=False, mask_identity=False, highlevel=True, behavior=None, attrs=None)

Counts an array’s elements over one or all levels of nesting.

Many types are supported, including all Awkward Arrays and Records. The
identity of counting is `0` and it is usually not masked.

This function has no analog in NumPy because counting values in a
rectilinear array would only result in elements of the NumPy array’s
[shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html).

The gaps and None values are not counted, and if a None value occurs at
a higher axis than the one being counted, it is kept as a placeholder
so that the outer list length does not change.

See [`ak.sum`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) for a more complete description of nested list and missing
value (None) handling in reducers.

If it is desirable to include None values in [`ak.count`](sphinx-llm:3ed114887b2c44a097b9f2699672fba0), use [`ak.fill_none`](sphinx-llm:e26b7c2675674008bc03fc5f18da53c5#ak.fill_none)
to turn the None values into something that would be counted.

If it is desirable to exclude NaN (“not a number”) values from [`ak.count`](sphinx-llm:3ed114887b2c44a097b9f2699672fba0),
use [`ak.nan_to_none`](sphinx-llm:9ecf4aa6eb61432b9955ca347e71f9ff#ak.nan_to_none) to turn them into None, which are not counted.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – If None, combine all values from the array into
    a single scalar result; if an int, group by that axis: `0` is the
    outermost, `1` is the first level of nested lists, etc., and
    negative `axis` counts from the innermost: `-1` is the innermost,
    `-2` is the next level up, etc; if a str, it is interpreted as the
    name of the axis which maps to an int if named axes are present.
    Named axes are attached to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and
    removed with [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If False, this reducer decreases the number of
    dimensions by 1; if True, the reduced values are wrapped in a new
    length-1 dimension so that the result of this operation may be
    broadcasted with the original array.
  * **mask_identity** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, reducing over empty lists results in
    None (an option type); otherwise, reducing over empty lists
    results in the operation’s identity.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  The number of elements of `array`.

### Examples

However, for nested lists of variable dimension and missing values, the
result of counting is non-trivial. For example, with this

```pycon
>>> array = ak.Array([[ 0.1,  0.2      ],
...                   [None, 10.2, None],
...                   None,
...                   [20.1, 20.2, 20.3],
...                   [30.1, 30.2      ]])
```

the result of counting over the innermost dimension is

```pycon
>>> ak.count(array, axis=-1)
<Array [2, 1, None, 3, 2] type='5 * ?int64'>
```

the outermost dimension is

```pycon
>>> ak.count(array, axis=0)
<Array [3, 4, 1] type='3 * int64'>
```

and all dimensions is

```pycon
>>> ak.count(array, axis=None)
8
```

Note also that this function is different from [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num), which counts
the number of values at a given depth, maintaining structure: [`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num)
never counts across different lists the way that reducers do ([`ak.num`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num)
is not a reducer; [`ak.count`](sphinx-llm:3ed114887b2c44a097b9f2699672fba0) is). For the same `array`,

```pycon
>>> ak.num(array, axis=0)
5
>>> ak.num(array, axis=1)
<Array [2, 3, None, 3, 2] type='5 * ?int64'>
```
