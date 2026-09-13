# ak.transform

Defined in [awkward.operations.ak_transform](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_transform.py) on [line 26](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_transform.py#L26).

#### ak.transform(transformation, array, \*more_arrays, depth_context=None, lateral_context=None, allow_records=True, broadcast_parameters_rule='intersect', left_broadcast=True, right_broadcast=True, numpy_to_regular=False, regular_to_jagged=False, return_value='simplified', expect_return_value=False, highlevel=True, behavior=None, attrs=None)

Applies a transformation function to every node of one or more arrays.

This is a public interface to the infrastructure that is used to implement
most Awkward Array operations. As such, it’s very powerful, but low-level.

The primary purpose of this function is to allow you to edit one level of
structure without having to worry about what it’s embedded in. Suppose, for
instance, you want to apply NumPy’s `np.round` function to numerical data,
regardless of what lists or other structures they’re embedded in.

The return value must be a subclass of [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) (to replace the
array node) or None (to leave the array node unchanged).

If you pass multiple arrays to this function (`more_arrays`), those arrays
will be broadcasted and all inputs, at the same level of depth and structure,
will be passed to the `transformation` function as a group.

* **Parameters:**
  * **transformation** (*callable*) – Function to apply to each node of the array.
    See below for details.
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes), but not an
    [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record) or [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record).
  * **more_arrays** – Additional arrays to be broadcasted together (with first `array`)
    and used together in the transformation. See below for details.
  * **depth_context** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – User data to propagate through the transformation.
    New data added to `depth_context` is available to the entire *subtree*
    at which it is added, but no other *subtrees*. For example, data added
    during the transformation will not be in the original `depth_context`
    after the transformation.
  * **lateral_context** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – User data to propagate through the transformation.
    New data added to `lateral_context` is available at any later step of
    the depth-first walk over the tree, including *other subtrees*. For
    example, data added during the transformation will be in the original
    `lateral_context` after the transformation.
  * **allow_records** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If False and the recursive walk encounters any
    [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) nodes, an error is raised.
  * **broadcast_parameters_rule** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Rule for broadcasting parameters, one of:
    - `"intersect"`
    - `"all_or_nothing"`
    - `"one_to_one"`
    - `"none"`
  * **left_broadcast** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If `more_arrays` are provided, the parameter
    determines whether the arrays are left-broadcasted, which is
    Awkward-like broadcasting.
  * **right_broadcast** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If `more_arrays` are provided, the parameter
    determines whether the arrays are right-broadcasted, which is
    NumPy-like broadcasting.
  * **numpy_to_regular** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, multidimensional [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
    nodes are converted into [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) nodes before
    calling `transformation`.
  * **regular_to_jagged** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, regular-type lists are converted into
    variable-length lists before calling `transformation`.
  * **return_value** (`"none"`, `"original", ``"simplified"`) – If `"none"`, the output of
    this function is None; if `"original"`, untouched nodes surrounding
    the ones replaced by the `transformation` are returned in their original
    state; if `"simplified"`, the `ak.Content.simplified` constructor is
    used on the surrounding nodes to ensure that option-type and union-type
    nodes are not nested inappropriately. Note that if `return_value` is `"none"`,
    the only way to get information out of this function is through the
    `lateral_context`.
  * **expect_return_value** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, raise a `RuntimeError` if the transformer
    does not terminate the recursion.
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  The array (or arrays) obtained by applying `transformation` to every node —
  either a transformed copy, or data extracted from a walk over the arrays’
  low-level layout nodes.

### Examples

Here is a “hello world” example:

```pycon
>>> def say_hello(layout, depth, **kwargs):
...     print("Hello", type(layout).__name__, "at", depth)
...
>>> array = ak.Array([[1.1, 2.2, "three"], [], None, [4.4, 5.5]])
>>> ak.transform(say_hello, array, return_value="none")
Hello IndexedOptionArray at 1
Hello ListOffsetArray at 1
Hello UnionArray at 2
Hello NumpyArray at 2
Hello ListOffsetArray at 2
Hello NumpyArray at 3
```

In the above, `say_hello` is called on every node of the `array`, which has
a lot of nodes because it has nested lists, missing data, and a union of
different types. The data types are low-level “layouts,” subclasses of
[`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), rather than high-level [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array).

```pycon
>>> def rounder(layout, **kwargs):
...     if layout.is_numpy:
...         return ak.contents.NumpyArray(
...             np.round(layout.data).astype(np.int32)
...         )
...
>>> array = ak.Array(
... [[[[[1.1, 2.2, 3.3], []], None], []],
...  [[[[4.4, 5.5]]]]]
... )
>>> ak.transform(rounder, array).show(type=True)
type: 2 * var * var * option[var * var * int32]
[[[[[1, 2, 3], []], None], []],
 [[[[4, 6]]]]]
```

Here is an example with broadcasting:

```pycon
>>> def combine(layouts, **kwargs):
...     assert len(layouts) == 2
...     if layouts[0].is_numpy and layouts[1].is_numpy:
...         return ak.contents.NumpyArray(
...             layouts[0].data + 10 * layouts[1].data
...         )
...
>>> array1 = ak.Array([[1, 2, 3], [], None, [4, 5]])
>>> array2 = ak.Array([1, 2, 3, 4])
>>> ak.transform(combine, array1, array2)
<Array [[11, 12, 13], [], None, [44, 45]] type='4 * option[var * int64]'>
```

The `1` and `4` from `array2` are broadcasted to the `[1, 2, 3]` and the
`[4, 5]` of `array1`, and the other elements disappear because they are
broadcasted with an empty list and a missing value. Note that the first argument
of this `transformation` function is a *list* of layouts, not a single layout.
There are always 2 layouts because 2 arrays were passed to [`ak.transform`](sphinx-llm:6e75f57f56094e488b36dd1c00b14970).

## Signature of the transformation function

If there is only one array, the first argument of `transformation` is a
[`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) instance. If there are multiple arrays (`more_arrays`),
the first argument is a list of [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) instances.

All other arguments can be absorbed into a `**kwargs` because they will always
be passed to your function by keyword. They are

* depth (int): The current list depth, where 1 is the outermost array and
  : higher numbers are deeper levels of list nesting. This does not count
    nesting of other data structures, such as option-types and records.
* depth_context (None or dict): Any user-specified data. You can add to
  : this dict during transformation; changes would only be seen in the
    subtree’s nodes.
* lateral_context (None or dict): Any user-specified data. You can add to
  : this dict during transformation; changes would be seen in any node
    visited later in the depth-first search.
* continuation (callable): Zero-argument function that continues the
  : recursion from this point in the walk, so that you can perform
    post-processing instead of pre-processing.

For completeness, the following arguments are also passed to `transformation`,
but you usually won’t need them:

* behavior (None or dict): Behavior that would be attached to the output
  : array(s) if `highlevel`.
* backend (array library / kernel library shim): Handle to the NumPy
  : library, CuPy, etc., depending on the type of arrays.
* options (dict): Options provided to [`ak.transform`](sphinx-llm:6e75f57f56094e488b36dd1c00b14970).

If there is only one array, the `transformation` function must either return
None or return an [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content).

If there are multiple arrays (`more_arrays`), then the transformation function
may return one array or a tuple of arrays. (The preferred type is a tuple, even
if it has length 1.)

The final return value of [`ak.transform`](sphinx-llm:6e75f57f56094e488b36dd1c00b14970) is a new array or tuple of arrays
constructed by replacing nodes when `transformation` returns a
[`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) or tuple of [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content), and leaving
nodes unchanged when `transformation` returns None. If `transformation` returns
length-1 tuples, the final output is an array, not a length-1 tuple.

If `return_value` is `"none"`, [`ak.transform`](sphinx-llm:6e75f57f56094e488b36dd1c00b14970) returns None. This is useful for
functions that return non-array data through `lateral_context`. The other two
choices, `"original"` and `"simplified"`, determine how untouched array nodes,
the ones that are \_not_ modified by the `transformation` function, are returned.
With `"original"`, they are returned without modification, which might result
in illegal combinations of option-type and union-type, which would raise an
error. With `"simplified"`, the surrounding array nodes are simplified upon
reconstruction. For example, if the `transformation` puts a new [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray)
inside an existing [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray), the two will be consolidated
into a single option-type array node.

## Contexts

The `depth_context` and `lateral_context` allow you to pass your own data into
the transformation as well as communicate between calls of `transformation` on
different nodes. The `depth_context` limits this communication to descendants
of the subtree in which the data were added; `lateral_context` does not have
this limit. (`depth_context` is shallow-copied at each node during descent;
`lateral_context` is never copied.)

For example, consider this array:

```pycon
>>> array = ak.Array([
...     [{"x": [1], "y": 1.1}, {"x": [1, 2], "y": 2.2}, {"x": [1, 2, 3], "y": 3.3}],
...     [],
...     [{"x": [1, 2, 3, 4], "y": 4.4}, {"x": [1, 2, 3, 4, 5], "y": 5.5}],
... ])
```

If we accumulate node type names using `depth_context`,

```pycon
>>> def crawl(layout, depth_context, **kwargs):
...     depth_context["types"] = depth_context["types"] + (type(layout).__name__,)
...     print(depth_context["types"])
...
>>> context = {"types": ()}
>>> ak.transform(crawl, array, depth_context=context, return_value="none")
('ListOffsetArray',)
('ListOffsetArray', 'RecordArray')
('ListOffsetArray', 'RecordArray', 'ListOffsetArray')
('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray')
('ListOffsetArray', 'RecordArray', 'NumpyArray')
>>> context
{'types': ()}
```

The data in `depth_context["types"]` represents a path from the root of the
tree to the current node. There is never, for instance, more than one leaf-type
([`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)) in the tuple. Also, the `context` is unchanged
outside of the function.

On the other hand, if we do the same with a `lateral_context`,

```pycon
>>> def crawl(layout, lateral_context, **kwargs):
...     lateral_context["types"] = lateral_context["types"] + (type(layout).__name__,)
...     print(lateral_context["types"])
...
>>> context = {"types": ()}
>>> ak.transform(crawl, array, lateral_context=context, return_value="none")
('ListOffsetArray',)
('ListOffsetArray', 'RecordArray')
('ListOffsetArray', 'RecordArray', 'ListOffsetArray')
('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray')
('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray', 'NumpyArray')
>>> context
{'types': ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray', 'NumpyArray')}
```

The data accumulate through the walk over the tree. There are two leaf-types
([`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)) in the tuple because this tree has two leaves.
The data are even available outside of the function, so `lateral_context` can
be paired with `return_value="none"` to extract non-array data, rather than
transforming the array.

The visitation order is stable: a recursive walk always proceeds through the
same tree in the same order.

## Continuation

The `transformation` function is given an input, untransformed layout or layouts.
Some algorithms need to perform a correction on transformed outputs, so
`continuation()` can be called at any point to continue descending but obtain
the transformed result.

For example, this function inserts an option-type at every level of an array:

```pycon
>>> def insert_optiontype(layout, continuation, **kwargs):
...     return ak.contents.UnmaskedArray(continuation())
...
>>> array = ak.Array([[[[[1.1, 2.2, 3.3], []]], []], [[[[4.4, 5.5]]]]])
>>> array.type.show()
2 * var * var * var * var * float64
```

```pycon
>>> array2 = ak.transform(insert_optiontype, array)
>>> array2.type.show()
2 * option[var * option[var * option[var * option[var * ?float64]]]]
```

In the original array, every node is a [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) except
the leaf, which is a [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray). The call to `continuation()`
returns a [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) with its contents transformed, which
is the argument of a new [`ak.contents.UnmaskedArray`](sphinx-llm:c65ee37a79b9404bbe625ef99e63bade#ak.contents.UnmaskedArray).

To see this process as it happens, we can add `print` statements to the function.

```pycon
>>> def insert_optiontype(input, continuation, **kwargs):
...     print("before", input.form.type)
...     output = ak.contents.UnmaskedArray(continuation())
...     print("after ", output.form.type)
...     return output
...
>>> ak.transform(insert_optiontype, array)
before var * var * var * var * float64
before var * var * var * float64
before var * var * float64
before var * float64
before float64
after  ?float64
after  option[var * ?float64]
after  option[var * option[var * ?float64]]
after  option[var * option[var * option[var * ?float64]]]
after  option[var * option[var * option[var * option[var * ?float64]]]]
<Array [[[[[1.1, ..., 3.3], ...]], ...], ...] type='2 * option[var * option...'>
```

## Broadcasting

When multiple arrays are provided (`more_arrays`), all of the arrays are
broadcasted during the walk so that the `transformation` function is eventually
provided with a list of layouts that have compatible types (for mathematical
operations, etc.).

For instance, given these two arrays:

```pycon
>>> array1 = ak.Array([[1, 2, 3], [], None, [4, 5]])
>>> array2 = ak.Array([10, 20, 30, 40])
```

The following single-array function shows the nodes encountered when walking
down either one of them.

```pycon
>>> def one_array(layout, **kwargs):
...     print(type(layout).__name__)
...
>>> ak.transform(one_array, array1, return_value="none")
IndexedOptionArray
ListOffsetArray
NumpyArray
>>> ak.transform(one_array, array2, return_value="none")
NumpyArray
```

The first array has three nested nodes; the second has only one node.

However, when the following two-array function is applied,

> ```pycon
> >>> def two_arrays(layouts, **kwargs):
> ...     assert len(layouts) == 2
> ...     print(type(layouts[0]).__name__, ak.to_list(layouts[0]))
> ...     print(type(layouts[1]).__name__, ak.to_list(layouts[1]))
> ...     print()
> ...
> >>> ak.transform(two_arrays, array1, array2)
> RegularArray [[[1, 2, 3], [], None, [4, 5]]]
> RegularArray [[10, 20, 30, 40]]
> ```

> IndexedOptionArray [[1, 2, 3], [], None, [4, 5]]
> NumpyArray [10, 20, 30, 40]

> ListArray [[1, 2, 3], [], [4, 5]]
> NumpyArray [10, 20, 40]

> NumpyArray [1, 2, 3, 4, 5]
> NumpyArray [10, 10, 10, 40, 40]

> (<Array [[1, 2, 3], [], None, [4, 5]] type=’4 \* option[var \* int64]’>,
> : <Array [[10, 10, 10], [], None, [40, 40]] type=’4 \* option[var \* int64]’>)

The incompatible types of the two arrays eventually becomes the same type by
duplicating and removing values wherever necessary. If you cannot perform an
operation on a [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) and a [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray),
wait for a later iteration, in which both will be [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray)
(if the original arrays are broadcastable).

The return value, without transformation, is the same as what
[`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays) would return. See [`ak.broadcast_arrays`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays) for an
explanation of `left_broadcast` and `right_broadcast`.

## Broadcasting Parameters

When broadcasting multiple arrays with parameters, there are different ways of
assigning parameters to the outputs. The assignment of array parameters happens
at every level above the transformation action.

The method of parameter assignment used by the broadcasting routine is controlled
by the `broadcast_parameters_rule` option, which can take one of the following
values:

`"intersect"`
: The parameters of each output array will correspond to the intersection
  of the parameters from each of the input arrays.

`"all_or_nothing"`
: If the parameters of the input arrays are all equal, then they will be used
  for each output array. Otherwise, the output arrays will not be given
  parameters.

`"one_to_one"`
: If the number of output arrays matches the number of input arrays, then the
  output arrays are given the parameters of the input arrays. Otherwise, a
  ValueError is raised.

`"none"`
: The output arrays will not be given parameters.

## Performance Tip

[`ak.transform`](sphinx-llm:6e75f57f56094e488b36dd1c00b14970) will traverse the layout of (potentially multiple) arrays once.
This can be useful if one wants to apply a batch of transformations in one single
layout traversal. Traversing the layout multiple times can be inefficient.

Consider the following example:

```pycon
>>> def batch_of_operations(array):
...     return np.sqrt(np.sin(array) + 1) - 1
...
>>> def apply_batch_of_operations(layout, **kwargs):
...     if layout.is_numpy:
...         return ak.contents.NumpyArray(
...             batch_of_operations(layout.data)
...         )
...
>>> array = ak.Array(
... [[[[[1.1, 2.2, 3.3], []], None], []],
...  [[[[4.4, 5.5]]]]]
... )
>>> %timeit ak.transform(apply_batch_of_operations, array)
... 68.5 μs ± 663 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
>>> %timeit batch_of_operations(array)
... 1.07 ms ± 39.1 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

The first `%timeit` cell shows the time it takes to apply the batch of operations using [`ak.transform`](sphinx-llm:6e75f57f56094e488b36dd1c00b14970),
which allows to apply the operations in one single traversal of the layout. The second `%timeit` cell shows
the runtime of applying the operations directly to the array, which traverses the layout multiple times.
To be more explicit: one layout traversal for each operation.

See also: [`ak.is_valid`](sphinx-llm:cb72d589ba0e433d9cedf3f12cc0da1c#ak.is_valid) and `ak.valid_when` to check the validity of transformed
outputs.
