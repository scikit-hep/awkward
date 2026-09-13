# How to reduce dimensions (sum/min/any/all)

After elementwise functions, dimension-reducer functions are the most commonly used. These functions replace a list of numbers with a single, scalar number by adding, multiplying, minimizing, maximizing, or performing logical-or (“any”) or logical-and (“all”).

These are also called aggregation functions; in relational databases, SQL, and data-frames, aggregations are applied after a “group by” operation. Awkward Array doesn’t have “group by” operations; lists are already grouped.

```ipython3
import awkward as ak
import numpy as np
```

## First reducer: `ak.sum`

To illustrate all of these functions, let’s consider addition. Given an array:

```ipython3
array = ak.Array([[1, 2, 3], [4, 5], [], [6]])
```

[`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) with no arguments adds all of the values in the nested lists, just like `np.sum`.

```ipython3
ak.sum(array)
```

With Awkward Arrays, it’s usually more useful to supply an `axis` argument to reduce one dimension, rather than all dimensions.

For reasons that will be explained below, `axis=-1` is the most frequently useful.

```ipython3
ak.sum(array, axis=-1)
```

### The `axis` argument

Before getting deeper into the `axis` argument, let’s consider a NumPy array with more dimensions.

```ipython3
array3d = np.array([
    [
        [    1,     2,     3,     4,     5],
        [   10,    20,    30,    40,    50],
        [  100,   200,   300,   400,   500],
    ],
    [
        [0.1  , 0.2  , 0.3  , 0.4  , 0.5  ],
        [0.01 , 0.02 , 0.03 , 0.04 , 0.05 ],
        [0.001, 0.002, 0.003, 0.004, 0.005],
    ],
])

with np.printoptions(suppress=True):
    print(array3d)
```

```myst-ansi
[[[  1.      2.      3.      4.      5.   ]
  [ 10.     20.     30.     40.     50.   ]
  [100.    200.    300.    400.    500.   ]]

 [[  0.1     0.2     0.3     0.4     0.5  ]
  [  0.01    0.02    0.03    0.04    0.05 ]
  [  0.001   0.002   0.003   0.004   0.005]]]
```

This array has 3 dimensions, so in addition to `axis=None` (reduce everything to a scalar), there are 3 possible axis values.

The first case, `axis=0`, adds the first 3×5 block to the second 3×5 block, i.e. summing over the first (length-2) dimension. Thus, the `1` is added to `0.1`, the `2` is added to `0.2`, and so on until the `500` is added to `0.005`.

```ipython3
with np.printoptions(suppress=True):
    print(np.sum(array3d, axis=0))
```

```myst-ansi
[[  1.1     2.2     3.3     4.4     5.5  ]
 [ 10.01   20.02   30.03   40.04   50.05 ]
 [100.001 200.002 300.003 400.004 500.005]]
```

The second case, `axis=1`, adds vertically within each 3×5 block, i.e. summing over the second (length-3) dimension. What’s left are two lists of length 5.

```ipython3
with np.printoptions(suppress=True):
    print(np.sum(array3d, axis=1))
```

```myst-ansi
[[111.    222.    333.    444.    555.   ]
 [  0.111   0.222   0.333   0.444   0.555]]
```

The third case, `axis=2`, adds horizontally within each 3×5 block, i.e. summing over the third (length-5) dimension. What’s left are two lists of length 3.

```ipython3
with np.printoptions(suppress=True):
    print(np.sum(array3d, axis=2))
```

```myst-ansi
[[  15.     150.    1500.   ]
 [   1.5      0.15     0.015]]
```

Since negative `axis` counts from the other end of the scale,

* `axis=0` is equivalent to `axis=-3`
* `axis=1` is equivalent to `axis=-2`
* `axis=2` is equivalent to `axis=-1`.

### The `axis` argument with ragged lists

Awkward Arrays allow the lengths of lists in an array to differ, so we can have

```ipython3
array_ragged = ak.Array([
    [  1,   2,   3     ],
    [ 10,  20          ],
    [100, 200, 300, 400],
])
array_ragged
```

As before, `axis=-1` sums over the innermost lists, replacing each of the 3 horizontal rows with a sum.

```ipython3
ak.sum(array_ragged, axis=-1)
```

And `axis=-2` sums vertically, replacing each of the 4 vertical columns with a sum. Since the list lengths differ, some of the places we might expect to see a value is an empty gap—it contributes nothing to the result.

```ipython3
ak.sum(array_ragged, axis=0)
```

We also have to choose a convention: should the values be left-aligned or right-aligned within their lists? Awkward Array choses left-aligned.

In ragged data from real datasets, summing over whole lists usually has more meaning than summing over parts of different lists, so `axis=-1` is usually the most meaningful choice of `axis`.

### The `axis` argument with missing data

Just as empty gaps contribute nothing to the sum, missing values (`None`) don’t contribute anything, either.

```ipython3
array_ragged = ak.Array([
    [None, None,    3,    4],
    [  10, None,   30      ],
    [ 100,  200,  300,  400],
])
array_ragged
```

`axis=-1` sums over each inner list, horizontally, replacing it with a scalar.

```ipython3
ak.sum(array_ragged, axis=-1)
```

And `axis=-2` sums over the outer dimension, vertically.

```ipython3
ak.sum(array_ragged, axis=-2)
```

For [`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum), each `None` has the same effect as a `0` value, for [`ak.prod()`](sphinx-llm:22420dea392a4bf0893e0034c465bc34#ak.prod) (multiplication), each `None` has the same effect as a `1` value, etc.

## The `keepdims` argument

Sometimes, you want to replace lists with a length-1 list, rather than a scalar. `keepdims=True` does that.

```ipython3
ak.sum(array_ragged, axis=-1, keepdims=True)
```

```ipython3
ak.sum(array_ragged, axis=-2, keepdims=True)
```

The `keepdims` argument is particularly useful for [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax), which return positions in a list where the value is minimized or maximized. Those positions can only be used as slice indexes if they’re at the right nesting level, which `keepdims=True` maintains.

## Other reducers

* The [`ak.prod()`](sphinx-llm:22420dea392a4bf0893e0034c465bc34#ak.prod) reducer multiplies, rather than adding.
* [`ak.min()`](sphinx-llm:d40aa9e4100d4897a291c711bc7a7042#ak.min) and [`ak.max()`](sphinx-llm:8d04431ada5f4197aee6b08d9430b68f#ak.max) minimize and maximize, returning `None` for empty lists.
* [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) return the index positions of the minimum or maximum value, with `None` for empty lists.
* [`ak.nansum()`](sphinx-llm:ff60aa82e6d241c4a55328326f918cf3#ak.nansum), [`ak.nanprod()`](sphinx-llm:4182c8d481c34fb4984c0c8a59234e00#ak.nanprod), [`ak.nanmin()`](sphinx-llm:2747d92f436d4b9f9ddb9465cefd98d6#ak.nanmin), [`ak.nanmax()`](sphinx-llm:0496eb0d118a4f93a16b1f74771eae82#ak.nanmax), [`ak.nanargmin()`](sphinx-llm:a9535e38ef0042b8bdf70e6523d74697#ak.nanargmin), and [`ak.nanargmax()`](sphinx-llm:c683af619d8149c3826ff22b161507d4#ak.nanargmax) ignore floating-point `nan` values before operating, the way that all reducers ignore `None` values before operating.
* [`ak.count_nonzero()`](sphinx-llm:8deedef4d9884c96bb016cd87c7f679d#ak.count_nonzero) counts non-zero values.
* [`ak.count()`](sphinx-llm:bc7e7716292a4cefb6c9ba610d4c76b6#ak.count) simply counts values. In NumPy, there’s no need for such a function because it would return constants (drawn from the NumPy array’s `shape`), but for ragged arrays, it counts the number of values that enter into a reduction. [`ak.num()`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num) also returns lengths of lists, but in a way that’s more useful for slicing; [`ak.count()`](sphinx-llm:bc7e7716292a4cefb6c9ba610d4c76b6#ak.count) is useful as the denominator of expressions in which another reducer (with the same `axis` and `keepdims` choices) is in the numerator.
* [`ak.any()`](sphinx-llm:df7c468b43ca4558ab741d771ffbf0dc#ak.any) and [`ak.all()`](sphinx-llm:6d63d32372de42f69a3faba14948813c#ak.all) reduce like logical-or and logical-and, which makes them particularly useful in slices (below).

## Reducing over “any” and “all”

[`ak.any()`](sphinx-llm:df7c468b43ca4558ab741d771ffbf0dc#ak.any) and [`ak.all()`](sphinx-llm:6d63d32372de42f69a3faba14948813c#ak.all) reduce boolean arrays, asking if a predicate is satisfied by “any” item or “all” items, respectively.

```ipython3
array_bool = ak.Array([
    [False, False,  True,  True],
    [False,  True, False,  True],
    [False,  True,  True,  True],
])
array_bool
```

```ipython3
ak.any(array_bool, axis=-1)
```

```ipython3
ak.any(array_bool, axis=-2)
```

```ipython3
ak.all(array_bool, axis=-1)
```

```ipython3
ak.all(array_bool, axis=-2)
```

Since logical-or is like addition of booleans and logical-and is like multiplication, these reducers could have been replaced with [`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) and [`ak.prod()`](sphinx-llm:22420dea392a4bf0893e0034c465bc34#ak.prod), but they’re very useful to have because they make some boolean-array slices easier to read.

```ipython3
array = ak.Array([[0, 1, 2], [], [-3, 4], [-5], [-6, -7, -8, -9]])
array
```

Select *whole lists* if *any* of their values are negative:

```ipython3
array[ak.any(array < 0, axis=-1)]
```

Select *whole lists* if *all* of their values are negative:

```ipython3
array[ak.all(array < 0, axis=-1)]
```

(If a list is empty, all of its elements satisfy a constraint.)

In both cases above, the selection can be read like an English sentence, “select lists if *any*…” or “select lists if *all*…”.

## Heterogeneous data and records cannot be reduced

These two kinds of data types are not reducible. Heterogeneous data allows an array to have multiple numbers of dimensions, so the problem is ill-posed:

```ipython3
ak.sum(ak.Array([[1.1, 2.2, 3.3], [], 4.4, 5.5]))
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[25], line 1
----> 1 ak.sum(ak.Array([[1.1, 2.2, 3.3], [], 4.4, 5.5]))

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_sum.py:224, in sum(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)
    221 yield (array,)
    223 # Implementation
--> 224 return _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_sum.py:317, in _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs, dtype)
    313 axis = regularize_axis(axis, none_allowed=True)
    315 reducer = ak._reducers.Sum(dtype=dtype)
--> 317 out = ak._do.reduce(
    318     layout,
    319     reducer,
    320     axis=axis,
    321     mask=mask_identity,
    322     keepdims=keepdims,
    323     behavior=ctx.behavior,
    324 )
    326 wrapped_out = ctx.wrap(
    327     out,
    328     highlevel=highlevel,
    329     allow_other=True,
    330 )
    332 # propagate named axis to output

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_do.py:252, in reduce(layout, reducer, axis, mask, keepdims, behavior)
    240 parts = remove_structure(
    241     layout,
    242     flatten_records=False,
   (...)    246     list_to_regular=True,
    247 )
    249 if len(parts) > 1:
    250     # We know that `flatten_records` must fail, so the only other type
    251     # that can return multiple parts here is the union array
--> 252     raise ValueError(
    253         "cannot use axis=None on an array containing irreducible unions"
    254     )
    255 elif len(parts) == 0:
    256     layout = ak.contents.EmptyArray()

ValueError: cannot use axis=None on an array containing irreducible unions

This error occurred while calling

    ak.sum(
        <Array [[1.1, 2.2, 3.3], [], 4.4, 5.5] type='4 * union[var * float6...'>
    )
```

And records are sometimes used to represent data with coordinates; applying [`ak.sum()`](sphinx-llm:08d8d4556ebd40868b7762f4e0ae9d1a#ak.sum) to non-Cartesian coordinates would be a subtle error.

```ipython3
ak.sum(ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}]), axis=-1)
```

```ipythontb
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[26], line 1
----> 1 ak.sum(ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}]), axis=-1)

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_sum.py:224, in sum(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)
    221 yield (array,)
    223 # Implementation
--> 224 return _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_sum.py:317, in _impl(array, axis, keepdims, mask_identity, highlevel, behavior, attrs, dtype)
    313 axis = regularize_axis(axis, none_allowed=True)
    315 reducer = ak._reducers.Sum(dtype=dtype)
--> 317 out = ak._do.reduce(
    318     layout,
    319     reducer,
    320     axis=axis,
    321     mask=mask_identity,
    322     keepdims=keepdims,
    323     behavior=ctx.behavior,
    324 )
    326 wrapped_out = ctx.wrap(
    327     out,
    328     highlevel=highlevel,
    329     allow_other=True,
    330 )
    332 # propagate named axis to output

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_do.py:328, in reduce(layout, reducer, axis, mask, keepdims, behavior)
    325 offsets = ak.index.Index64([0, layout.length], nplike=layout.backend.nplike)
    327 shifts = None
--> 328 next = layout._reduce_next(
    329     reducer,
    330     negaxis,
    331     starts,
    332     shifts,
    333     offsets,
    334     1,
    335     mask,
    336     keepdims,
    337     behavior,
    338 )
    340 return next[0]

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/recordarray.py:960, in RecordArray._reduce_next(self, reducer, negaxis, starts, shifts, offsets, outlength, mask, keepdims, behavior)
    958 reducer_recordclass = find_record_reducer(reducer, self, behavior)
    959 if reducer_recordclass is None:
--> 960     raise TypeError(
    961         "no ak.{} overloads for custom types: {}".format(
    962             reducer.name, ", ".join(self.fields)
    963         )
    964     )
    965 else:
    966     # Positional reducers ultimately need to do more work when rebuilding the result
    967     # so asking for a mask doesn't help us!
    968     reducer_should_mask = mask and not reducer.needs_position

TypeError: no ak.sum overloads for custom types: x, y

This error occurred while calling

    ak.sum(
        <Array [{x: 1.1, y: [1]}, {...}] type='2 * {x: float64, y: var * in...'>
        axis = -1
    )
```
