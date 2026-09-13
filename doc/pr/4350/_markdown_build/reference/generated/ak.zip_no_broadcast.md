# ak.zip_no_broadcast

Defined in [awkward.operations.ak_zip_no_broadcast](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_zip_no_broadcast.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_zip_no_broadcast.py#L19).

#### ak.zip_no_broadcast(arrays, \*, parameters=None, with_name=None, highlevel=True, behavior=None, attrs=None)

Combines arrays into a collection of records or tuples without broadcasting.

Caution: unlike [`ak.zip`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) this function will \_not_ broadcast the arrays
together. During typetracing, it assumes that the given arrays have already
the same layouts and lengths.

This operation may be thought of as the opposite of projection in
[`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__), which extracts fields one at a time, or [`ak.unzip`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip),
which extracts them all in one call.

See also [`ak.zip`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) and [`ak.unzip`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip).

* **Parameters:**
  * **arrays** (*mapping* *or* *sequence* *of* *arrays*) – Each value in this mapping or
    sequence can be any array-like data that [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes.
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
  An array of records (or tuples) whose fields (or slots) are the
  `arrays`, combined into a single structure.

### Examples

Consider the following arrays, `one` and `two`.

```pycon
>>> one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]])
>>> two = ak.Array([["a", "b", "c"], [], ["d", "e"], ["f"]])
```

Zipping them together using a dict creates a collection of records with
the same nesting structure as `one` and `two`.

```pycon
>>> ak.zip_no_broadcast({"x": one, "y": two}).show()
[[{x: 1.1, y: 'a'}, {x: 2.2, y: 'b'}, {x: 3.3, y: 'c'}],
 [],
 [{x: 4.4, y: 'd'}],
 []]
```

Doing so with a list creates tuples, whose fields are not named.

```pycon
>>> ak.zip_no_broadcast([one, two]).show()
[[(1.1, 'a'), (2.2, 'b'), (3.3, 'c')],
 [],
 [(4.4, 'd')],
 []]
```
