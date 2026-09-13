# ak.zip

Defined in [awkward.operations.ak_zip](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_zip.py) on [line 19](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_zip.py#L19).

#### ak.zip(arrays, depth_limit=None, \*, parameters=None, with_name=None, right_broadcast=False, optiontype_outside_record=False, highlevel=True, behavior=None, attrs=None)

Combines arrays into records or tuples, broadcasting them together.

If the `arrays` have nested structure, they are broadcasted with one
another to form the records or tuples as deeply as possible, though this
can be limited by `depth_limit`.

This operation may be thought of as the opposite of projection in
[`ak.Array.__getitem__`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.__getitem__), which extracts fields one at a time, or [`ak.unzip`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip),
which extracts them all in one call.

As an extreme, `depth_limit=1` is a handy way to make a record structure
at the outermost level, regardless of whether the fields have matching
structure or not.

* **Parameters:**
  * **arrays** (*mapping* *or* *sequence* *of* *arrays*) – Each value in this mapping or
    sequence can be any array-like data that [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes.
  * **depth_limit** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int)) – If None, attempt to fully broadcast the
    `array` to all levels. If an int, limit the number of dimensions
    that get broadcasted. The minimum value is `1`, for no
    broadcasting.
  * **parameters** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Parameters for the new
    [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node that is created by this operation.
  * **with_name** (*None* *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Assigns a `"__record__"` name to the new
    [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node that is created by this operation
    (overriding `parameters`, if necessary).
  * **right_broadcast** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, follow rules for implicit
    right-broadcasting, as described in [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays).
  * **optiontype_outside_record** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, continue broadcasting past
    any option types before creating the new [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node.
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
>>> ak.zip({"x": one, "y": two}).show()
[[{x: 1.1, y: 'a'}, {x: 2.2, y: 'b'}, {x: 3.3, y: 'c'}],
 [],
 [{x: 4.4, y: 'd'}, {x: 5.5, y: 'e'}],
 [{x: 6.6, y: 'f'}]]
```

Doing so with a list creates tuples, whose fields are not named.

```pycon
>>> ak.zip([one, two]).show()
[[(1.1, 'a'), (2.2, 'b'), (3.3, 'c')],
 [],
 [(4.4, 'd'), (5.5, 'e')],
 [(6.6, 'f')]]
```

Adding a third array with the same length as `one` and `two` but less
internal structure is okay: it gets broadcasted to match the others.
(See [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays) for broadcasting rules.)

```pycon
>>> three = ak.Array([100, 200, 300, 400])
>>> ak.zip([one, two, three]).show()
[[(1.1, 'a', 100), (2.2, 'b', 100), (3.3, 'c', 100)],
 [],
 [(4.4, 'd', 300), (5.5, 'e', 300)],
 [(6.6, 'f', 400)]]
```

However, if arrays have the same depth but different lengths of nested
lists, attempting to zip them together is a broadcasting error.

```pycon
>>> one = ak.Array([[[1, 2, 3], [], [4, 5], [6]], [], [[7, 8]]])
>>> two = ak.Array([[[1.1, 2.2], [3.3], [4.4], [5.5]], [], [[6.6]]])
>>> ak.zip([one, two])
ValueError: while calling
    ak.zip(
        arrays = [<Array [[[1, 2, 3], [], [4, ...], [6]], ...] type='3 * var ...
        depth_limit = None
        parameters = None
        with_name = None
        right_broadcast = False
        optiontype_outside_record = False
        highlevel = True
        behavior = None
    )
Error details: cannot broadcast nested list
```

For this, one can set the `depth_limit` to prevent the operation from
attempting to broadcast what can’t be broadcasted.

```pycon
>>> ak.zip([one, two], depth_limit=1).show()
[([[1, 2, 3], [], [4, ...], [6]], [[1.1, ...], ...]),
 ([], []),
 ([[7, 8]], [[6.6]])]
```

When zipping together arrays with optional values, it can be useful to create
the [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) node after the option types. By default, [`ak.zip`](sphinx-llm:05f20ea68bd54975a8fd2acd52e20334)
does not do this:

```pycon
>>> one = ak.Array([1, 2, None])
>>> two = ak.Array([None, 5, 6])
>>> ak.zip([one, two])
<Array [(1, None), (2, 5), (None, 6)] type='3 * (?int64, ?int64)'>
```

If the `optiontype_outside_record` option is set to `True`, Awkward will continue to
broadcast the arrays together at the depth_limit until it reaches non-option
types. This effectively takes the union of the option mask:

```pycon
>>> ak.zip([one, two], optiontype_outside_record=True)
<Array [None, (2, 5), None] type='3 * ?(int64, int64)'>
```
