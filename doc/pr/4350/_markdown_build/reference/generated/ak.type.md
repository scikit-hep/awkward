# ak.type

Defined in [awkward.operations.ak_type](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_type.py) on [line 20](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_type.py#L20).

#### ak.type(array, \*, behavior=None)

Returns the high-level type of an array as a Type object.

The high-level type ignores layout differences like [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray)
versus [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray), but not differences like
“regular-sized lists” (i.e. [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray)) versus
“variable-sized lists” (i.e. [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) and similar).

Types are rendered as [Datashape](https://datashape.readthedocs.io/)
strings, which makes the same distinctions.

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output type, if
    high-level.
* **Returns:**
  The high-level type of an `array` (many types supported, including all
  Awkward Arrays and Records) as [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type) objects.

### Examples

For example,

```pycon
>>> array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
...                   [],
...                   [{"x": 3.3, "y": [3, 3, 3]}]])
```

has type

```pycon
>>> ak.type(array).show()
3 * var * {
    x: float64,
    y: var * int64
}
```

but

```pycon
>>> array = ak.Array(np.arange(2*3*5).reshape(2, 3, 5))
```

has type

```pycon
>>> ak.type(array).show()
2 * 3 * 5 * int64
```

Some cases, like heterogeneous data, require [extensions beyond the
Datashape specification]([https://github.com/blaze/datashape/issues/237](https://github.com/blaze/datashape/issues/237)).
For example,

```pycon
>>> array = ak.Array([1, "two", [3, 3, 3]])
```

has type

```pycon
>>> ak.type(array).show()
3 * union[
    int64,
    string,
    var * int64
]
```

but “union” is not a Datashape type-constructor. (Its syntax is
similar to existing type-constructors, so it’s a plausible addition
to the language.)
