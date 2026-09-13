# ak.ravel

Defined in [awkward.operations.ak_ravel](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_ravel.py) on [line 16](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_ravel.py#L16).

#### ak.ravel(array, \*, highlevel=True, behavior=None, attrs=None)

Returns an array with all levels of nesting removed.

Missing values are not eliminated by flattening. See [`ak.flatten`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) with
`axis=None` for an equivalent function that eliminates the option type.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with all level of nesting removed by erasing the
  boundaries between consecutive lists.

  This is the equivalent of NumPy’s `np.ravel` for Awkward Arrays.

### Examples

Consider the following:

```pycon
>>> array = ak.Array([[[1.1, 2.2, 3.3],
...                    [],
...                    [4.4, 5.5],
...                    [6.6]],
...                   [],
...                   [[7.7],
...                    [8.8, 9.9]
...                   ]])
```

Ravelling the array produces a flat array

```pycon
>>> ak.ravel(array).show()
[1.1,
 2.2,
 3.3,
 4.4,
 5.5,
 6.6,
 7.7,
 8.8,
 9.9]
```
