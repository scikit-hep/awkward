# How to flatten arrays, especially for plotting

In a data analysis, it is important to plot your data frequently, and the interactive nature of array-at-a-time functions facilitate that.

However, plotting views your data as a generic set or sequence—the structure of nested lists and records can’t be captured by standard plots. Histograms (including 2-dimensional heatmaps) take input data to be an unordered set, as do scatter plots. Connected-line plots, such as time-series, use the sequential order of the data, but there aren’t many visualizations that show nestedness. (Maybe there will be, in the future.)

As such, these standard plotting routines expect simple structures, either a single flat array (in which the order may be relevant or irrelevant) or several same-length arrays (in which the relative or absolute order is relevant). Encountering an Awkward Array, they may try to call `np.asarray` on it, which only works if the array can be made rectilinear or they may try to iterate over it in Python, which can be prohibitively slow if the dataset is large.

## Scope of destructuring

To destructure an array for plotting, you’ll want to

* remove nested lists, definitely for variable-length ones (”`var *`” in the type string) and possibly for regular ones as well (”`N *`” in the type string, where `N` is an integer),
* remove record structures,
* remove missing data

There are two functions that are responsible for flattening arrays: [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) with `axis=None`; and [`ak.ravel()`](sphinx-llm:790180d227054cc4a11cade4d8ccbc68#ak.ravel); but you don’t want to apply them without thinking, because structure is important to the meaning of your data and you want to be able to interpret the plot. Destructuring is an information-losing operation, so your guidance is required to eliminate exactly the structure you want to eliminate, and there are several ways to do that, depending on what you want to do.

After destructuring, you might *still* need to call `np.asarray` on the output because the plotting library might not recognize an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) as an array. You’ll probably also want to develop your destructuring on a commandline or a different Jupyter cell from the plotting library function call, to understand what structure the output has without the added complication of the plotting library’s error messages.

```ipython3
import awkward as ak
import numpy as np
```

## ak.ravel

First, let’s create an array with some interesting structure.

```ipython3
array = ak.Array(
    [[{"x": 1.1, "y": [1]}, {"x": None, "y": [1, 2]}], [], [{"x": 3.3, "y": [1, 2, 3]}]]
)
array
```

As mentioned above, [`ak.ravel()`](sphinx-llm:790180d227054cc4a11cade4d8ccbc68#ak.ravel) is one of two functions that turns any array into a 1-dimensional array with no nested lists, no nested records.

```ipython3
ak.ravel(array)
```

Calling this function on an already flat array does nothing, so you don’t have to worry about what state your array had been in before you called it.

```ipython3
ak.ravel(ak.ravel(array))
```

Unlike `ak.flatten(..., axis=None)`, [`ak.ravel()`](sphinx-llm:790180d227054cc4a11cade4d8ccbc68#ak.ravel) preserves [`None`](https://docs.python.org/3/library/constants.html#None) values at the leaves, meaning that functions which expect a simple array of numbers will usually raise an exception.

However, there are a few questions you should be asking yourself:

* Did the nested lists have special meaning? What does the plot represent if I just concatenate them all?
* Did the record fields have distinct meanings? In this example, what does it mean to put floating-point *x* values and nested-list *y* values in the same bucket of numbers to plot? Does it matter that there are more *y* values than *x* values? **In most circumstances, you do not want to mix record fields in a plot.**

## ak.flatten with axis=None

If [`ak.ravel()`](sphinx-llm:790180d227054cc4a11cade4d8ccbc68#ak.ravel) is a sledgehammer, then [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) with `axis=None` is a pile driver that turns any array into a 1-dimensional array with no nested lists, no nested records, and no missing data.

```ipython3
array = ak.Array(
    [[{"x": 1.1, "y": [1]}, {"x": None, "y": [1, 2]}], [], [{"x": 3.3, "y": [1, 2, 3]}]]
)
array
```

```ipython3
ak.flatten(array, axis=None)
```

Like [`ak.ravel()`](sphinx-llm:790180d227054cc4a11cade4d8ccbc68#ak.ravel), Calling this function on an already flat array does nothing, so you don’t have to worry about what state your array had been in before you called it.

```ipython3
ak.flatten(ak.flatten(array, axis=None), axis=None)
```

In addition to the concerns raised above, it is also important to consider whether the [`None`](https://docs.python.org/3/library/constants.html#None) values in your array are meaningful. For example, consider an array of x-axis and y-axis values. If only the y-axis contains [`None`](https://docs.python.org/3/library/constants.html#None) values, `ak.flatten(y_values, axis=None)` would produce an array that does not align with the flattened x-axis values.

```ipython3
x = ak.Array([[1, 2, 3], [4, 5, 6, 7]])
y = ak.Array([[8, None, 6], [5, None, None, 4]])

z = 2 * np.ravel(x) + np.ravel(y)
```

## Selecting record fields

A more controlled way to extract fields from a record is to [project]() them by name.

```ipython3
array = ak.Array(
    [
        [{"x": 1.1, "y": [1], "z": "one"}, {"x": None, "y": [1, 2], "z": "two"}],
        [],
        [{"x": 3.3, "y": [1, 2, 3], "z": "three"}],
    ]
)
array
```

If we want only the *x* field, we can ask for it as an attribute (because it’s a valid Python name) or with a string-valued slice:

```ipython3
array.x
```

```ipython3
array["x"]
```

This controls the biggest deficiency of [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) with `axis=None`, the mixing of data with different meanings.

```ipython3
ak.flatten(array.x, axis=None)
```

```ipython3
ak.flatten(array.y, axis=None)
```

If some of your fields can be safely flattened—together into one set—and others can’t, you can use a list of strings to pick just the fields you want.

```ipython3
ak.flatten(array[["x", "y"]], axis=None)
```

(Careful! A tuple has a special meaning in slices, which doesn’t apply here.)

```ipython3
array[("x", "y")]
```

```ipythontb
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[15], line 1
----> 1 array[("x", "y")]

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1118, in Array.__getitem__(self, where)
    689 def __getitem__(self, where):
    690     """
    691     Args:
    692         where (many types supported; see below): Index of positions to
   (...)   1116     have the same dimension as the array being indexed.
   1117     """
-> 1118     with ak._errors.SlicingErrorContext(self, where):
   1119         # Handle named axis
   1120         (_, ndim) = self._layout.minmax_depth
   1121         named_axis = _get_named_axis(self)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1126, in Array.__getitem__(self, where)
   1122 where = _normalize_named_slice(named_axis, where, ndim)
   1124 NamedAxis.mapping = named_axis
-> 1126 indexed_layout = prepare_layout(self._layout._getitem(where, NamedAxis))
   1128 if NamedAxis.mapping:
   1129     return ak.operations.ak_with_named_axis._impl(
   1130         indexed_layout,
   1131         named_axis=NamedAxis.mapping,
   (...)   1134         attrs=self._attrs,
   1135     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:651, in Content._getitem(self, where, named_axis)
    642 named_axis.mapping = _named_axis
    644 next = ak.contents.RegularArray(
    645     this,
    646     this.length,
    647     1,
    648     parameters=None,
    649 )
--> 651 out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
    653 if out.length is not unknown_length and out.length == 0:
    654     return out._getitem_nothing()

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/regulararray.py:625, in RegularArray._getitem_next(self, head, tail, advanced)
    617         return RegularArray(
    618             nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
    619             nextsize,
    620             self.length,
    621             parameters=self._parameters,
    622         )
    624 elif isinstance(head, str):
--> 625     return self._getitem_next_field(head, tail, advanced)
    627 elif isinstance(head, list):
    628     return self._getitem_next_fields(head, tail, advanced)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:328, in Content._getitem_next_field(self, head, tail, advanced)
    321 def _getitem_next_field(
    322     self,
    323     head: SliceItem | tuple,
    324     tail: tuple[SliceItem, ...],
    325     advanced: Index | None,
    326 ):
    327     nexthead, nexttail = ak._slicing.head_tail(tail)
--> 328     return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/regulararray.py:625, in RegularArray._getitem_next(self, head, tail, advanced)
    617         return RegularArray(
    618             nextcontent._getitem_next(nexthead, nexttail, nextadvanced),
    619             nextsize,
    620             self.length,
    621             parameters=self._parameters,
    622         )
    624 elif isinstance(head, str):
--> 625     return self._getitem_next_field(head, tail, advanced)
    627 elif isinstance(head, list):
    628     return self._getitem_next_fields(head, tail, advanced)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:328, in Content._getitem_next_field(self, head, tail, advanced)
    321 def _getitem_next_field(
    322     self,
    323     head: SliceItem | tuple,
    324     tail: tuple[SliceItem, ...],
    325     advanced: Index | None,
    326 ):
    327     nexthead, nexttail = ak._slicing.head_tail(tail)
--> 328     return self._getitem_field(head)._getitem_next(nexthead, nexttail, advanced)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/regulararray.py:393, in RegularArray._getitem_field(self, where, only_fields)
    389 def _getitem_field(
    390     self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    391 ) -> Content:
    392     return RegularArray(
--> 393         self._content._getitem_field(where, only_fields),
    394         self._size,
    395         self._length,
    396         self._zeros_length_generator,
    397         parameters=None,
    398     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listoffsetarray.py:367, in ListOffsetArray._getitem_field(self, where, only_fields)
    362 def _getitem_field(
    363     self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    364 ) -> Content:
    365     return ListOffsetArray(
    366         self._offsets,
--> 367         self._content._getitem_field(where, only_fields),
    368         parameters=None,
    369     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/indexedoptionarray.py:375, in IndexedOptionArray._getitem_field(self, where, only_fields)
    370 def _getitem_field(
    371     self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    372 ) -> Content:
    373     return IndexedOptionArray.simplified(
    374         self._index,
--> 375         self._content._getitem_field(where, only_fields),
    376         parameters=None,
    377     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/numpyarray.py:358, in NumpyArray._getitem_field(self, where, only_fields)
    355 def _getitem_field(
    356     self, where: str | SupportsIndex, only_fields: tuple[str, ...] = ()
    357 ) -> Content:
--> 358     raise ak._errors.index_error(self, where, "not an array of records")

IndexError: cannot slice NumpyArray (of length 2) with 'y': not an array of records

This error occurred while attempting to slice

    <Array [[{x: 1.1, y: [1], ...}, ...], ...] type='3 * var * {x: ?float64...'>

with

    ('x', 'y')
```

If you have records inside of records, you can extract them with [nested projection]() if they have common names.

```ipython3
array = ak.Array(
    [
        {"x": {"up": 1, "down": -1}, "y": {"up": 1.1, "down": -1.1}},
        {"x": {"up": 2, "down": -2}, "y": {"up": 2.2, "down": -2.2}},
        {"x": {"up": 3, "down": -3}, "y": {"up": 3.3, "down": -3.3}},
        {"x": {"up": 4, "down": -4}, "y": {"up": 4.4, "down": -4.4}},
    ]
)
array
```

```ipython3
ak.flatten(array[["x", "y"], "up"], axis=None)
```

## ak.flatten for one axis

Since `axis=None` is so dangerous, the default value of [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) is `axis=1`. This flattens only the first nested dimension.

```ipython3
ak.flatten(ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]]))
```

It also removes missing values *in the axis that is being flattened* because flattening considers a missing list like an empty list.

```ipython3
ak.flatten(ak.Array([[0, 1, 2], None, [3, 4], [5], [6, 7, 8, 9]]))
```

It does not flatten or remove missing values from any other axis.

```ipython3
ak.flatten(ak.Array([[[0, 1, 2, 3, 4]], [], [[5], [6, 7, 8, 9]]]))
```

```ipython3
ak.flatten(ak.Array([[[0, 1, 2, None]], [], [[5], [6, 7, 8, 9]]]))
```

Moreover, you can’t flatten already-flat data because a 1-dimensional array does not have an `axis=1`. (`axis` starts counting at `0`.)

```ipython3
ak.flatten(ak.Array([1, 2, 3, 4, 5]))
```

```ipythontb
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
Cell In[22], line 1
----> 1 ak.flatten(ak.Array([1, 2, 3, 4, 5]))

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_dispatch.py:40, in named_high_level_function.<locals>.dispatch(*args, **kwargs)
     37 @wraps(func)
     38 def dispatch(*args, **kwargs):
     39     # NOTE: this decorator assumes that the operation is exposed under `ak.`
---> 40     with OperationErrorContext(name, args, kwargs):
     41         gen_or_result = func(*args, **kwargs)
     42         if isgenerator(gen_or_result):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_dispatch.py:66, in named_high_level_function.<locals>.dispatch(*args, **kwargs)
     64 # Failed to find a custom overload, so resume the original function
     65 try:
---> 66     next(gen_or_result)
     67 except StopIteration as err:
     68     return err.value

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_flatten.py:185, in flatten(array, axis, highlevel, behavior, attrs)
    182 yield (array,)
    184 # Implementation
--> 185 return _impl(array, axis, highlevel, behavior, attrs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_flatten.py:266, in _impl(array, axis, highlevel, behavior, attrs)
    264     out = apply(layout)
    265 else:
--> 266     out = ak._do.flatten(layout, axis)
    268 wrapped_out = ctx.wrap(
    269     out,
    270     highlevel=highlevel,
    271 )
    273 # propagate named axis to output
    274 #   if axis == None: use strategy "remove all" (see: awkward._namedaxis)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_do.py:195, in flatten(layout, axis)
    194 def flatten(layout: Content, axis: int = 1) -> Content:
--> 195     _offsets, flattened = layout._offsets_and_flattened(axis, 1)
    196     return flattened

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/numpyarray.py:471, in NumpyArray._offsets_and_flattened(self, axis, depth)
    468     return self.to_RegularArray()._offsets_and_flattened(axis, depth)
    470 else:
--> 471     raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")

AxisError: axis=1 exceeds the depth of this array (1)

This error occurred while calling

    ak.flatten(
        <Array [1, 2, 3, 4, 5] type='5 * int64'>
    )
```

`axis=0` is a valid option for [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten), but since there can’t be any lists at this level, it only removes missing values.

```ipython3
ak.flatten(ak.Array([1, 2, 3, None, None, 4, 5]), axis=0)
```

## Selecting one element from each list

Flattening removes list structure without removing values. Often, you want to do the opposite of that: you want to plot one element from each list. This makes the plot “aware” of your list structure.

This kind of operation is usually just a slice.

```ipython3
array = ak.Array([[0, 1, 2], [3, 4], [5], [6, 7, 8, 9]])
array
```

```ipython3
array[:, 0]
```

The above syntax selects all lists from the array (`axis=0`) and the first element from each list (`axis=1`). We could have as easily selected the last:

```ipython3
array[:, -1]
```

A plot made from `ak.flatten(array)` would be a plot of all numbers with no knowledge of lists; a plot made from `array[:, 0]` would be a plot of lists, as represented by the first element in each. It depends on what you want to plot.

What if you get this error?

```ipython3
array = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array
```

```ipython3
array[:, 0]
```

```ipythontb
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[28], line 1
----> 1 array[:, 0]

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1118, in Array.__getitem__(self, where)
    689 def __getitem__(self, where):
    690     """
    691     Args:
    692         where (many types supported; see below): Index of positions to
   (...)   1116     have the same dimension as the array being indexed.
   1117     """
-> 1118     with ak._errors.SlicingErrorContext(self, where):
   1119         # Handle named axis
   1120         (_, ndim) = self._layout.minmax_depth
   1121         named_axis = _get_named_axis(self)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1126, in Array.__getitem__(self, where)
   1122 where = _normalize_named_slice(named_axis, where, ndim)
   1124 NamedAxis.mapping = named_axis
-> 1126 indexed_layout = prepare_layout(self._layout._getitem(where, NamedAxis))
   1128 if NamedAxis.mapping:
   1129     return ak.operations.ak_with_named_axis._impl(
   1130         indexed_layout,
   1131         named_axis=NamedAxis.mapping,
   (...)   1134         attrs=self._attrs,
   1135     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:651, in Content._getitem(self, where, named_axis)
    642 named_axis.mapping = _named_axis
    644 next = ak.contents.RegularArray(
    645     this,
    646     this.length,
    647     1,
    648     parameters=None,
    649 )
--> 651 out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
    653 if out.length is not unknown_length and out.length == 0:
    654     return out._getitem_nothing()

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/regulararray.py:595, in RegularArray._getitem_next(self, head, tail, advanced)
    589 nextcontent = self._content._carry(nextcarry, True)
    591 if advanced is None or (
    592     advanced.length is not unknown_length and advanced.length == 0
    593 ):
    594     return RegularArray(
--> 595         nextcontent._getitem_next(nexthead, nexttail, advanced),
    596         nextsize,
    597         self.length,
    598         parameters=self._parameters,
    599     )
    600 else:
    601     nextadvanced = ak.index.Index64.empty(nextcarry.length, nplike)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listarray.py:764, in ListArray._getitem_next(self, head, tail, advanced)
    758 head = ak._slicing.normalize_integer_like(head)
    759 assert (
    760     nextcarry.nplike is self._backend.nplike
    761     and self._starts.nplike is self._backend.nplike
    762     and self._stops.nplike is self._backend.nplike
    763 )
--> 764 self._maybe_index_error(
    765     self._backend[
    766         "awkward_ListArray_getitem_next_at",
    767         nextcarry.dtype.type,
    768         self._starts.dtype.type,
    769         self._stops.dtype.type,
    770     ](
    771         nextcarry.data,
    772         self._starts.data,
    773         self._stops.data,
    774         lenstarts,
    775         head,
    776     ),
    777     slicer=head,
    778 )
    779 nextcontent = self._content._carry(nextcarry, True)
    780 return nextcontent._getitem_next(nexthead, nexttail, advanced)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:297, in Content._maybe_index_error(self, error, slicer)
    295 else:
    296     message = self._backend.format_kernel_error(error)
--> 297     raise ak._errors.index_error(self, slicer, message)

IndexError: cannot slice ListArray (of length 5) with array(0): index out of range while attempting to get index 0 (in compiled code: https://github.com/scikit-hep/awkward/blob/awkward-cpp-56/awkward-cpp/src/cpu-kernels/awkward_ListArray_getitem_next_at.cpp#L21)

This error occurred while attempting to slice

    <Array [[0, 1, 2], [], ..., [5], [6, 7, 8, 9]] type='5 * var * int64'>

with

    (:, 0)
```

It says that it can’t get element `0` of one of the lists, and that’s because this `array` contains an empty list.

One way to deal with that is to take a range-slice, rather than ask for an individual element from each list.

```ipython3
array[:, :1]
```

But this array still has structure, so you can flatten it *as an additional step*.

```ipython3
ak.flatten(array[:, :1])
```

Alternatively, you may want to attack the problem head-on: the issue is that some lists have too few elements, so why not remove those lists with an explicit slice? The [`ak.num()`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num) function tells us the length of each nested list.

```ipython3
ak.num(array)
```

```ipython3
ak.num(array) > 0
```

Slicing the first dimension with this would ensure that the second dimension always has the element we seek.

```ipython3
array[ak.num(array) > 0, 0]
```

The same applies if we’re taking the last element:

```ipython3
array[ak.num(array) > 0, -1]
```

You can also do fancy things, requesting both the first and last element of each list, as long as it doesn’t run afoul of slicing rules (which were constrained to match NumPy’s in cases that overlap).

```ipython3
array[
    ak.num(array) > 0, [0, -1]
]  # these two arrays have different lengths, can't be broadcasted as in NumPy advanced slicing
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[35], line 1
----> 1 array[
      2     ak.num(array) > 0, [0, -1]
      3 ]  # these two arrays have different lengths, can't be broadcasted as in NumPy advanced slicing

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1118, in Array.__getitem__(self, where)
    689 def __getitem__(self, where):
    690     """
    691     Args:
    692         where (many types supported; see below): Index of positions to
   (...)   1116     have the same dimension as the array being indexed.
   1117     """
-> 1118     with ak._errors.SlicingErrorContext(self, where):
   1119         # Handle named axis
   1120         (_, ndim) = self._layout.minmax_depth
   1121         named_axis = _get_named_axis(self)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1126, in Array.__getitem__(self, where)
   1122 where = _normalize_named_slice(named_axis, where, ndim)
   1124 NamedAxis.mapping = named_axis
-> 1126 indexed_layout = prepare_layout(self._layout._getitem(where, NamedAxis))
   1128 if NamedAxis.mapping:
   1129     return ak.operations.ak_with_named_axis._impl(
   1130         indexed_layout,
   1131         named_axis=NamedAxis.mapping,
   (...)   1134         attrs=self._attrs,
   1135     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:586, in Content._getitem(self, where, named_axis)
    584 items = ak._slicing.normalise_items(where, backend)
    585 # Prepare items for advanced indexing (e.g. via broadcasting)
--> 586 nextwhere = ak._slicing.prepare_advanced_indexing(items, backend)
    588 # Handle named axis
    589 # first expand the ellipsis to colons in nextwhere,
    590 # copy nextwhere to not pollute the original
    591 _nextwhere = tuple(nextwhere)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_slicing.py:120, in prepare_advanced_indexing(items, backend)
    118 # Then broadcast the index items
    119 nplike = backend.nplike
--> 120 broadcasted = nplike.broadcast_arrays(*[nplike.asarray(x) for x in broadcastable])
    122 # And re-assemble the index with the broadcasted items
    123 prepared = []

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_nplikes/array_module.py:336, in ArrayModuleNumpyLike.broadcast_arrays(self, *arrays)
    334 def broadcast_arrays(self, *arrays: ArrayLikeT) -> list[ArrayLikeT]:
    335     arrays = maybe_materialize(*arrays)
--> 336     return self._module.broadcast_arrays(*arrays)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/numpy/lib/_stride_tricks_impl.py:577, in broadcast_arrays(subok, *args)
    570 # nditer is not used here to avoid the limit of 64 arrays.
    571 # Otherwise, something like the following one-liner would suffice:
    572 # return np.nditer(args, flags=['multi_index', 'zerosize_ok'],
    573 #                  order='C').itviews
    575 args = [np.array(_m, copy=None, subok=subok) for _m in args]
--> 577 shape = _broadcast_shape(*args)
    579 result = [array if array.shape == shape
    580           else _broadcast_to(array, shape, subok=subok, readonly=False)
    581                           for array in args]
    582 return tuple(result)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/numpy/lib/_stride_tricks_impl.py:452, in _broadcast_shape(*args)
    447 """Returns the shape of the arrays that would result from broadcasting the
    448 supplied arrays against each other.
    449 """
    450 # use the old-iterator because np.nditer does not handle size 0 arrays
    451 # consistently
--> 452 b = np.broadcast(*args[:64])
    453 # unfortunately, it cannot handle 64 or more arguments directly
    454 for pos in range(64, len(args), 63):
    455     # ironically, np.broadcast does not properly handle np.broadcast
    456     # objects (it treats them as scalars)
    457     # use broadcasting to avoid allocating the full array

ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (5,) and arg 1 with shape (2,).

This error occurred while attempting to slice

    <Array [[0, 1, 2], [], ..., [5], [6, 7, 8, 9]] type='5 * var * int64'>

with

    (<Array [True, False, True, True, True] type='5 * bool'>, [0, -1])
```

```ipython3
array[ak.num(array) > 0][:, [0, -1]]  # so just put them in different slices
```

And then flatten the result (if necessary—the shape is regular; some plotting libraries would interpret it as a single set of numbers).

```ipython3
ak.flatten(array[ak.num(array) > 0][:, [0, -1]])
```

## Aggregating each list

Reductions should be familiar to users of SQL and Pandas; after grouping data by some quantity, one must apply some aggregating operation on each group to get one number for each group. The one-element slices of the previous section are like SQL’s `FIRST_VALUE` and `LAST_VALUE`, which is a special case of reducing.

The architypical aggregation function is “sum,” which reduces a list by adding up its values. [`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) and its relatives, [`ak.prod()`](sphinx-llm:22420dea392a4bf0893e0034c465bc34#ak.prod) (product/multiplication), [`ak.mean()`](sphinx-llm:9dfbdca7e4104165863b3b3f4391b21e#ak.mean), etc., are all reducers in Awkward Array.

Following NumPy, their default `axis` is `None`, but for this application, you’ll need to specify an explicit axis.

```ipython3
array = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array
```

```ipython3
ak.sum(array, axis=1)
```

Some of these are not defined for empty lists, so you’ll need to either replace the missing values with [`ak.fill_none()`](sphinx-llm:e26b7c2675674008bc03fc5f18da53c5#ak.fill_none) or flatten them.

```ipython3
ak.mean(array, axis=1)
```

```ipython3
ak.fill_none(ak.mean(array, axis=1), 0)  # fill with zero
```

```ipython3
ak.fill_none(ak.mean(array, axis=1), ak.mean(array))  # fill with the mean of all
```

```ipython3
ak.flatten(ak.mean(array, axis=1), axis=0)
```

Each of these has a different effect: filling with `0` puts an identifiable value in the plot (a peak at `0` if it’s a histogram), filling with the overall mean imputes a value in missing cases, flattening away the missing values reduces the number of entries in the plot. Each of these has a different meaning when interpreting your plot!

## Minimizing/maximizing over each list

Minimizing and maximizing are also reducers, [`ak.min()`](sphinx-llm:d40aa9e4100d4897a291c711bc7a7042#ak.min) and [`ak.max()`](sphinx-llm:8d04431ada5f4197aee6b08d9430b68f#ak.max) (and [`ak.ptp()`](sphinx-llm:8d4491031e5e4221b645d7275e922799#ak.ptp) for the peak-to-peak difference between the minimum and maximum).

They deserve their own section because they are an important case.

```ipython3
array = ak.Array([[0, 2, 1], [], [4, 3], [5], [8, 6, 7, 9]])
array
```

```ipython3
ak.min(array, axis=1)
```

```ipython3
ak.max(array, axis=1)
```

As before, they aren’t defined for empty lists, so you’ll have to *choose* a method to eliminate the missing values.

Sometimes, you want the “top N” elements from each list, rather than the “top 1.” Awkward Array doesn’t ([yet](https://github.com/scikit-hep/awkward-1.0/issues/554)) have a function for the “top N” elements, but it can be done with [`ak.sort()`](sphinx-llm:04e24ad5c9624efeab7a13c95c4f71cf#ak.sort) and a slice.

```ipython3
ak.sort(array, axis=1)
```

```ipython3
ak.sort(array, axis=1)[:, -2:]
```

We still have work to do: some of these lists are shorter than the 2 elements we asked for. What should be done with them? Eliminate all lists with fewer than two elements?

```ipython3
ak.sort(array[ak.num(array) >= 2], axis=1)[:, -2:]
```

Or just concatenate everything so that we don’t lose the lists with only one value (`5` in this example)?

```ipython3
ak.flatten(ak.sort(array, axis=1)[:, -2:])
```

## Minimizing/maximizing lists of records

Unlike numbers, records do not have an ordering: you cannot call [`ak.min()`](sphinx-llm:d40aa9e4100d4897a291c711bc7a7042#ak.min) on an array of records. But usually, what you want to do instead is to find the minimum or maximum of some quantity calculated from the records and pick records (or record fields) from that.

```ipython3
array = ak.Array(
    [
        [
            {"x": 2, "y": 2, "z": 2.2},
            {"x": 1, "y": 1, "z": 1.1},
            {"x": 3, "y": 3, "z": 3.3},
        ],
        [],
        [{"x": 5, "y": 5, "z": 5.5}, {"x": 4, "y": 4, "z": 4.4}],
        [
            {"x": 7, "y": 7, "z": 7.7},
            {"x": 9, "y": 9, "z": 9.9},
            {"x": 8, "y": 8, "z": 8.8},
            {"x": 6, "y": 6, "z": 6.6},
        ],
    ]
)
array
```

The [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) functions return the integer index where the minimum or maximum of some numeric formula can be found.

```ipython3
np.sqrt(array.x**2 + array.y**2)
```

```ipython3
ak.argmax(np.sqrt(array.x**2 + array.y**2), axis=1)
```

These integer indexes can be used as slices if they don’t eliminate a dimension, which can be requested via `keepdims=True`. This makes a length-1 list for each reduced output.

```ipython3
maximize_by = ak.argmax(np.sqrt(array.x**2 + array.y**2), axis=1, keepdims=True)
maximize_by
```

Applying this to the original `array`, we get the “best” record in each list, according to `maximize_by`.

```ipython3
array[maximize_by]
```

```ipython3
array[maximize_by].to_list()
```

This still has list structures and missing values, so it’s ready for [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten), assuming that we extract the appropriate record field to plot.

```ipython3
ak.flatten(array[maximize_by].z, axis=None)
```

## Concatenating independently restructured arrays

Sometimes, what you want to do can’t be a single expression. Suppose we have this data:

```ipython3
array = ak.Array(
    [[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}], [], [{"x": 3.3, "y": [1, 2, 3]}]]
)
array
```

and we want to combine all *x* values and the maximum *y* value in a plot. This requires a different expression on `array.x` from `array.y`.

```ipython3
ak.flatten(array.x)
```

```ipython3
ak.flatten(ak.max(array.y, axis=2), axis=None)
```

To get all of these into one array (because the plotting function only accepts one argument), you’ll need to [`ak.concatenate()`](sphinx-llm:9d688ff77559406881cf7aa931110693#ak.concatenate) them.

```ipython3
ak.concatenate(
    [
        ak.flatten(array.x),
        ak.flatten(ak.max(array.y, axis=2), axis=None),
    ]
)
```

## Maintaining alignment between arrays with missing values

Dropping missing values with [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) doesn’t keep track of where they were removed. This is a problem if the plotting library takes separate sequences for the x-axis and y-axis, and these must be aligned.

Instead of [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten), you can use [`ak.is_none()`](sphinx-llm:488667031383491ba3420edcd0c7126a#ak.is_none).

```ipython3
array = ak.Array(
    [
        {"x": 1, "y": 5.5},
        {"x": 2, "y": 3.3},
        {"x": None, "y": 2.2},
        {"x": 4, "y": None},
        {"x": 5, "y": 1.1},
    ]
)
array
```

```ipython3
ak.is_none(array.x)
```

```ipython3
ak.is_none(array.y)
```

```ipython3
to_keep = ~(ak.is_none(array.x) | ak.is_none(array.y))
to_keep
```

```ipython3
array.x[to_keep], array.y[to_keep]
```

## Actually drawing structure

If need be, you can change the plotter to match the data.

```ipython3
array = ak.Array(
    [
        [{"x": 1, "y": 3.3}, {"x": 2, "y": 1.1}, {"x": 3, "y": 2.2}],
        [],
        [{"x": 4, "y": 5.5}, {"x": 5, "y": 4.4}],
        [
            {"x": 5, "y": 1.1},
            {"x": 4, "y": 3.3},
            {"x": 2, "y": 5.5},
            {"x": 1, "y": 4.4},
        ],
    ]
)
array
```

```ipython3
import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches

fig, ax = plt.subplots()

for line in array:
    if len(line) > 0:
        vertices = np.dstack([np.asarray(line.x), np.asarray(line.y)])[0]
        codes = [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.LINETO] * (
            len(line) - 1
        )
        path = matplotlib.path.Path(vertices, codes)
        ax.add_patch(matplotlib.patches.PathPatch(path, facecolor="none"))

ax.set_xlim(0, 6)
ax.set_ylim(0, 6);
```

(The above example assumes that `len(array)` is small enough to iterate over in Python, but vectorizes over each list in the `array`. It was adapted from the [Matplotlib path tutorial](https://matplotlib.org/stable/tutorials/advanced/path_tutorial.html).)
