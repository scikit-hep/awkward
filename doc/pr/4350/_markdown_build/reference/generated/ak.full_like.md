# ak.full_like

Defined in [awkward.operations.ak_full_like](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_full_like.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_full_like.py#L19).

#### ak.full_like(array, fill_value, \*, dtype=None, including_unknown=False, highlevel=True, behavior=None, attrs=None)

Returns an array with the same structure as the input, filled with a given value.

This is the equivalent of NumPy’s `np.full_like` for Awkward Arrays.

Although it’s possible to produce an array of `fill_value` with the
structure of an `array` using [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays):

```pycon
>>> array = ak.Array([[1, 2, 3], [], [4, 5]])
>>> ak.broadcast_arrays(array, 1)
[<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>,
 <Array [[1, 1, 1], [], [1, 1]] type='3 * var * int64'>]
>>> ak.broadcast_arrays(array, 1.0)
[<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>,
 <Array [[1, 1, 1], [], [1, 1]] type='3 * var * float64'>]
```

Such a technique takes its type from the scalar (`1` or `1.0`), rather than
the array. This function gets all types from the array, which might not be
the same in all parts of the structure.

(There is no equivalent of NumPy’s `np.empty_like` because Awkward Arrays
are immutable.)

See also [`ak.zeros_like`](sphinx-llm:9876103fd1004198a431cc49602df858#ak.zeros_like) and [`ak.ones_like`](sphinx-llm:a6efac4195034ee78a35a71b042de1dd#ak.ones_like).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **fill_value** – Value to fill the new array with.
  * **dtype** (*None* *or* *NumPy dtype*) – Overrides the data type of the result.
  * **including_unknown** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, the `unknown` type is considered
    a value type and is converted to a zero-length array of the
    specified dtype; if False, `unknown` will remain `unknown`.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with the same structure as `array`, with every value replaced
  by `fill_value`.

### Examples

Here is an extreme example:

```pycon
>>> array = ak.Array([
... [{"x": 0.0, "y": []},
...  {"x": 1.1, "y": [1]},
...  {"x": 2.2, "y": [1, 2]}],
... [],
... [{"x": 3.3, "y": [1, 2, None, 3]},
...  False,
...  False,
...  True,
...  {"x": 4.4, "y": [1, 2, None, 3, 4]}]])
>>> ak.full_like(array, 12.3).show()
[[{x: 12.3, y: []}, {x: 12.3, y: [12]}, {x: 12.3, y: [12, 12]}],
 [],
 [{x: 12.3, y: [12, 12, None, 12]}, True, ..., True, {x: 12.3, y: [12, ...]}]]
```

The `"x"` values get filled in with `12.3` because they retain their type
(`float64`) and the `"y"` list items get filled in with `12` because they
retain their type (`int64`). Booleans get filled with True because `12.3`
is not zero. Missing values remain in the same positions as in the original
`array`. (To fill them in, use [`ak.fill_none`](sphinx-llm:e26b7c2675674008bc03fc5f18da53c5#ak.fill_none).)
