# ak.merge_union_of_records

Defined in [awkward.operations.ak_merge_union_of_records](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_merge_union_of_records.py) on [line 22](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_merge_union_of_records.py#L22).

#### ak.merge_union_of_records(array, axis=-1, \*, highlevel=True, behavior=None, attrs=None)

Simplifies unions of records into records of options.

For example, this turns

```pycon
>>> array = ak.concatenate(([{"a": 1}], [{"b": 2}]))
>>> array
<Array [{a: 1}, {b: 2}] type='2 * union[{a: int64}, {b: int64}]'>
```

into records of options, i.e.

```pycon
>>> ak.merge_union_of_records(array)
<Array [{a: 1, b: None}, {a: None, ...}] type='2 * {a: ?int64, b: ?int64}'>
```

Missing records are preserved in the result, e.g.

```pycon
>>> array = ak.concatenate(([{"a": 1}], [{"b": 2}, None]))
>>> array
<Array [{a: 1}, {b: 2}, None] type='3 * union[{a: int64}, ?{b: int64}]'>
>>> ak.merge_union_of_records(array)
<Array [{a: 1, b: None}, {...}, None] type='3 * ?{a: ?int64, b: ?int64}'>
```

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dimension at which this operation is applied.
    The outermost dimension is `0`, followed by `1`, etc., and negative
    values count backward from the  innermost: `-1` is the innermost
    dimension, `-2` is the next level up, etc.
    If a str, it is interpreted as the name of the axis which maps
    to an int if named axes are present. Named axes are attached
    to an array using [`ak.with_named_axis`](sphinx-llm:850fa57157f54411a7f413a32c8bd96a#ak.with_named_axis) and removed with
    [`ak.without_named_axis`](sphinx-llm:30b66a6c5ad241b0b374a553292778a2#ak.without_named_axis); also see the
    [Named axes user guide](../../user-guide/how-to-array-properties-named-axis.html).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An equivalent array with the union of records merged into a single record
  of options.
