# ak.mask

Defined in [awkward.operations.ak_mask](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_mask.py) on [line 18](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_mask.py#L18).

#### ak.mask(array, mask, \*, valid_when=True, highlevel=True, behavior=None, attrs=None)

Returns an array with elements replaced by None where a mask condition fails.

An array for which:

```default
output[i] = array[i] if mask[i] == valid_when else None
```

Unlike filtering data with [`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__), this `output` has the
same length as the original `array` and can therefore be used in
calculations with it, such as
[universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html).

Another syntax for:

```default
ak.mask(array, array_of_booleans)
```

is:

```default
array.mask[array_of_booleans]
```

(which is 5 characters away from simply filtering the `array`).

See [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays) for details about broadcasting and the generalized
set of broadcasting rules.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **mask** (*array* *of* *booleans*) – The mask that overlays elements in the
    `array` with None. Must have the same length as `array`.
  * **valid_when** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, True values in `mask` are considered
    valid (passed from `array` to the output); if False, False
    values in `mask` are considered valid.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with elements replaced by None where the mask condition fails.

### Examples

For example, with

```pycon
>>> array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

with a boolean selection of `good` elements like

```pycon
>>> good = (array % 2 == 1)
>>> good
<Array [False, True, False, True, ..., True, False, True] type='10 * bool'>
```

could be used to filter the original `array` (or another with the same
length).

```pycon
>>> array[good]
<Array [1, 3, 5, 7, 9] type='5 * int64'>
```

However, this eliminates information about which elements were dropped and
where they were. If we instead use [`ak.mask`](sphinx-llm:9055fbc4033e46b5b8a47ba5fcb0a8d8),

```pycon
>>> ak.mask(array, good)
<Array [None, 1, None, 3, None, 5, None, 7, None, 9] type='10 * ?int64'>
```

this information and the length of the array is preserved, and it can be
used in further calculations with the original `array` (or another with
the same length).

```pycon
>>> ak.mask(array, good) + array
<Array [None, 2, None, 6, None, 10, None, 14, None, 18] type='10 * ?int64'>
```

In particular, successive filters can be applied to the same array.

Even if the `array` and/or the `mask` is nested,

```pycon
>>> array = ak.Array([[[0, 1, 2], [], [3, 4], [5]], [[6, 7, 8], [9]]])
>>> good = (array % 2 == 1)
>>> good
<Array [[[False, True, False], ..., [True]], ...] type='2 * var * var * bool'>
```

it can still be used with [`ak.mask`](sphinx-llm:9055fbc4033e46b5b8a47ba5fcb0a8d8) because the `array` and `mask`
parameters are broadcasted.

```pycon
>>> ak.mask(array, good)
<Array [[[None, 1, None], [], ..., [5]], ...] type='2 * var * var * ?int64'>
```
