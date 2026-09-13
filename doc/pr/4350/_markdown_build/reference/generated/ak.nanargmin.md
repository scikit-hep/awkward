# ak.nanargmin

Defined in [awkward.operations.ak_argmin](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argmin.py) on [line 90](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_argmin.py#L90).

#### ak.nanargmin(array, axis=None, \*, keepdims=False, mask_identity=True, highlevel=True, behavior=None, attrs=None)

Returns the index of the minimum value, treating NaN values as missing.

Equivalent to:

```default
ak.argmin(ak.nan_to_none(array))
```

with all other arguments unchanged.

See also [`ak.argmin`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin).

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
  * **mask_identity** – If True, reducing over empty lists results in
    None (an option type); otherwise, reducing over empty lists
    results in the operation’s identity.
* **Returns:**
  Like [`ak.argmin`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin), but treating NaN (“not a number”) values as missing.
