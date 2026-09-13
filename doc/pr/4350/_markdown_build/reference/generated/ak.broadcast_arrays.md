# ak.broadcast_arrays

Defined in [awkward.operations.ak_broadcast_arrays](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_broadcast_arrays.py) on [line 28](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_broadcast_arrays.py#L28).

#### ak.broadcast_arrays(\*arrays, depth_limit=None, broadcast_parameters_rule='one_to_one', left_broadcast=True, right_broadcast=True, highlevel=True, behavior=None, attrs=None)

Broadcasts arrays together so they can be combined element-by-element.

One typically wants single-item-per-element data to be duplicated to
match multiple-items-per-element data. Operations on the broadcasted
arrays like:

```default
one_dimensional + nested_lists
```

would then have the same effect as the procedural code:

```default
for x, outer in zip(one_dimensional, nested_lists):
    output = []
    for inner in outer:
        output.append(x + inner)
    yield output
```

where `x` has the same value for each `inner` in the inner loop.

Awkward Array’s broadcasting manages to have it both ways by applying the
following rules:

* If all dimensions are regular (i.e. [`ak.types.RegularType`](sphinx-llm:a729849acf504584b9a7d4ae728c6263#ak.types.RegularType)), like NumPy,
  implicit broadcasting aligns to the right, like NumPy.
* If any dimension is variable (i.e. [`ak.types.ListType`](sphinx-llm:649f7f37622a4fbc999e70526300cf12#ak.types.ListType)), which can
  never be true of NumPy, implicit broadcasting aligns to the left.
* Explicit broadcasting with a length-1 regular dimension always
  broadcasts, like NumPy.

Thus, it is important to be aware of the distinction between a dimension
that is declared to be regular in the type specification and a dimension
that is allowed to be variable (even if it happens to have the same length
for all elements). This distinction is can be accessed through the
[`ak.Array.type`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.type), but it is lost when converting an array into JSON or
Python objects.

* **Parameters:**
  * **arrays** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **depth_limit** (*None* *or* [*int*](https://docs.python.org/3/library/functions.html#int) *,* *default is None*) – If None, attempt to fully
    broadcast the `arrays` to all levels. If an int, limit the number
    of dimensions that get broadcasted. The minimum value is `1`,
    for no broadcasting.
  * **broadcast_parameters_rule** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Rule for broadcasting parameters, one of:
    - `"intersect"`
    - `"all_or_nothing"`
    - `"one_to_one"`
    - `"none"`
  * **left_broadcast** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, follow rules for implicit
    left-broadcasting, as described below.
  * **right_broadcast** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, follow rules for implicit
    right-broadcasting, as described below.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default is True*) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array);
    otherwise, return a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  Like NumPy’s
  [broadcast_arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html)
  function, this function returns the input `arrays` with enough elements
  duplicated that they can be combined element-by-element.

  For NumPy arrays, this means that scalars are replaced with arrays with
  the same scalar value repeated at every element of the array, and regular
  dimensions are created to increase low-dimensional data into
  high-dimensional data.

### Examples

For example,

```pycon
>>> ak.broadcast_arrays(5,
...                     [1, 2, 3, 4, 5])
[<Array [5, 5, 5, 5, 5] type='5 * int64'>,
 <Array [1, 2, 3, 4, 5] type='5 * int64'>]
```

and

```pycon
>>> ak.broadcast_arrays(np.array([1, 2, 3]),
...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
[<Array [[  1,   2,   3], [ 1,  2,  3]] type='2 * 3 * int64'>,
 <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]
```

Note that in the second example, when the `3 * int64` array is expanded
to match the `2 * 3 * float64` array, it is the deepest dimension that
is aligned. If we try to match a `2 * int64` with the `2 * 3 * float64`,

```pycon
>>> ak.broadcast_arrays(np.array([1, 2]),
...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
ValueError: while calling
    ak.broadcast_arrays(
        arrays = (array([1, 2]), array([[ 0.1,  0.2,  0.3],
       [10. , 20....
        depth_limit = None
        broadcast_parameters_rule = 'one_to_one'
        left_broadcast = True
        right_broadcast = True
        highlevel = True
        behavior = None
    )
Error details: cannot broadcast RegularArray of size 2 with RegularArray of size 3
```

NumPy has the same behavior: arrays with different numbers of dimensions
are aligned to the right before expansion. One can control this by
explicitly adding a new axis (reshape to add a dimension of length 1)
where the expansion is supposed to take place because a dimension of
length 1 can be expanded like a scalar.

```pycon
>>> ak.broadcast_arrays(np.array([1, 2])[:, np.newaxis],
...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
[<Array [[  1,   1,   1], [ 2,  2,  2]] type='2 * 3 * int64'>,
 <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]
```

Again, NumPy does the same thing (`np.newaxis` is equal to None, so this
trick is often shown with None in the slice-tuple). Where the broadcasting
happens can be controlled, but numbers of dimensions that don’t match are
implicitly aligned to the right (fitting innermost structure, not
outermost).

While that might be an arbitrary decision for rectilinear arrays, it is
much more natural for implicit broadcasting to align left for tree-like
structures. That is, the root of each data structure should agree and
leaves may be duplicated to match. For example,

```pycon
>>> ak.broadcast_arrays([            100,   200,        300],
...                     [[1.1, 2.2, 3.3],    [], [4.4, 5.5]])
[<Array [[100, 100, 100], [], [300, 300]] type='3 * var * int64'>,
 <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>]
```

If arrays have the same depth but different lengths of nested
lists, attempting to broadcast them together is a broadcasting error.

```pycon
>>> one = ak.Array([[[1, 2, 3], [], [4, 5], [6]], [], [[7, 8]]])
>>> two = ak.Array([[[1.1, 2.2], [3.3], [4.4], [5.5]], [], [[6.6]]])
>>> ak.broadcast_arrays(one, two)
ValueError: while calling
    ak.broadcast_arrays(
        arrays = (<Array [[[1, 2, 3], [], [4, ...], [6]], ...] type='3 * var ...
        depth_limit = None
        broadcast_parameters_rule = 'one_to_one'
        left_broadcast = True
        right_broadcast = True
        highlevel = True
        behavior = None
    )
Error details: cannot broadcast nested list
```

For this, one can set the `depth_limit` to prevent the operation from
attempting to broadcast what can’t be broadcasted.

```pycon
>>> this, that = ak.broadcast_arrays(one, two, depth_limit=1)
>>> this.show()
[[[1, 2, 3], [], [4, 5], [6]],
 [],
 [[7, 8]]]
>>> that.show()
[[[1.1, 2.2], [3.3], [4.4], [5.5]],
 [],
 [[6.6]]]
```
