# How to filter arrays: cutting vs. masking

```ipython3
import awkward as ak
import numpy as np
```

## The problem with slicing

When you write a mathematical formula using binary operators like `+` and `*`, or [NumPy universal functions (ufuncs)](https://numpy.org/doc/stable/reference/ufuncs.html) like `np.sqrt`, the shapes of nested lists must align. If the arrays in an expression were derived from a single array, this is often automatic. For instance,

```ipython3
original_array = ak.Array([
    [
        {"title": "zero", "x": 0, "y": 0},
        {"title": "one", "x": 1, "y": 1.1},
        {"title": "two", "x": 2, "y": 2.2},
    ],
    [],
    [
        {"title": "three", "x": 3, "y": 3.3},
        {"title": "four", "x": 4, "y": 4.4},
    ],
    [
        {"title": "five", "x": 5, "y": 5.5},
    ],
    [
        {"title": "six", "x": 6, "y": 6.6},
        {"title": "seven", "x": 7, "y": 7.7},
        {"title": "eight", "x": 8, "y": 8.8},
        {"title": "nine", "x": 9, "y": 9.9},
    ],
])
```

```ipython3
array_x = original_array.x
array_y = original_array.y
```

The `array_x` and `array_y` have the same number of lists and the same numbers of items in each list because they were both slices of the `original_array`.

```ipython3
array_x
```

```ipython3
array_y
```

Thus, they can be used together in a mathematical formula.

```ipython3
array_x**2 + array_y**2
```

However, if one array is sliced, or if the two arrays are sliced by different criteria, they would no longer line up:

```ipython3
sliced_x = array_x[array_x > 3]
sliced_y = array_y[array_y > 3]
```

```ipython3
sliced_x
```

```ipython3
sliced_y
```

Notice that the first was sliced with `array_x > 3` and the second was sliced with `array_y > 3`, and as a result, the third list differs in length between the two arrays:

```ipython3
sliced_x[2], sliced_y[2]
```

If we try to use these together, we get a ValueError:

```ipython3
sliced_x**2 + sliced_y**2
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[11], line 1
----> 1 sliced_x**2 + sliced_y**2

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_operators.py:52, in _binary_method.<locals>.func(self, other)
     49 if _disables_array_ufunc(other):
     50     return NotImplemented
---> 52 return ufunc(self, other)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1644, in Array.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
   1579 """
   1580 Intercepts attempts to pass this Array to a NumPy
   1581 [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
   (...)   1641 See also #__array_function__.
   1642 """
   1643 name = f"{type(ufunc).__module__}.{ufunc.__name__}.{method!s}"
-> 1644 with ak._errors.OperationErrorContext(name, inputs, kwargs):
   1645     return ak._connect.numpy.array_ufunc(ufunc, method, inputs, kwargs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1645, in Array.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
   1643 name = f"{type(ufunc).__module__}.{ufunc.__name__}.{method!s}"
   1644 with ak._errors.OperationErrorContext(name, inputs, kwargs):
-> 1645     return ak._connect.numpy.array_ufunc(ufunc, method, inputs, kwargs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:484, in array_ufunc(ufunc, method, inputs, kwargs)
    476         raise TypeError(
    477             "no {}.{} overloads for custom types: {}".format(
    478                 type(ufunc).__module__, ufunc.__name__, ", ".join(error_message)
    479             )
    480         )
    482     return None
--> 484 out = ak._broadcasting.broadcast_and_apply(
    485     inputs,
    486     action,
    487     depth_context=depth_context,
    488     lateral_context=lateral_context,
    489     allow_records=False,
    490     function_name=ufunc.__name__,
    491 )
    493 out_named_axis = functools.reduce(
    494     _unify_named_axis, lateral_context[NAMED_AXIS_KEY].named_axis
    495 )
    496 if len(out) == 1:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1223, in broadcast_and_apply(inputs, action, depth_context, lateral_context, allow_records, left_broadcast, right_broadcast, numpy_to_regular, regular_to_jagged, function_name, broadcast_parameters_rule)
   1221 backend = backend_of(*inputs, coerce_to_common=False)
   1222 isscalar = []
-> 1223 out = apply_step(
   1224     backend,
   1225     broadcast_pack(inputs, isscalar),
   1226     action,
   1227     0,
   1228     depth_context,
   1229     lateral_context,
   1230     {
   1231         "allow_records": allow_records,
   1232         "left_broadcast": left_broadcast,
   1233         "right_broadcast": right_broadcast,
   1234         "numpy_to_regular": numpy_to_regular,
   1235         "regular_to_jagged": regular_to_jagged,
   1236         "function_name": function_name,
   1237         "broadcast_parameters_rule": broadcast_parameters_rule,
   1238     },
   1239 )
   1240 assert isinstance(out, tuple)
   1241 return tuple(broadcast_unpack(x, isscalar) for x in out)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1201, in apply_step(backend, inputs, action, depth, depth_context, lateral_context, options)
   1199     return result
   1200 elif result is None:
-> 1201     return continuation()
   1202 else:
   1203     raise AssertionError(result)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1170, in apply_step.<locals>.continuation()
   1168 # Any non-string list-types?
   1169 elif any(x.is_list and not is_string_like(x) for x in contents):
-> 1170     return broadcast_any_list()
   1172 # Any RecordArrays?
   1173 elif any(x.is_record for x in contents):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:663, in apply_step.<locals>.broadcast_any_list()
    660         nextinputs.append(x)
    661         nextparameters.append(NO_PARAMETERS)
--> 663 outcontent = apply_step(
    664     backend,
    665     nextinputs,
    666     action,
    667     depth + 1,
    668     copy.copy(depth_context),
    669     lateral_context,
    670     options,
    671 )
    672 assert isinstance(outcontent, tuple)
    673 parameters = parameters_factory(nextparameters, len(outcontent))

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1201, in apply_step(backend, inputs, action, depth, depth_context, lateral_context, options)
   1199     return result
   1200 elif result is None:
-> 1201     return continuation()
   1202 else:
   1203     raise AssertionError(result)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1170, in apply_step.<locals>.continuation()
   1168 # Any non-string list-types?
   1169 elif any(x.is_list and not is_string_like(x) for x in contents):
-> 1170     return broadcast_any_list()
   1172 # Any RecordArrays?
   1173 elif any(x.is_record for x in contents):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:714, in apply_step.<locals>.broadcast_any_list()
    710 for i, ((named_axis, ndim), x, x_is_string) in enumerate(
    711     zip(named_axes_with_ndims, inputs, input_is_string, strict=True)
    712 ):
    713     if isinstance(x, listtypes) and not x_is_string:
--> 714         next_content = broadcast_to_offsets_avoiding_carry(x, offsets)
    715         nextinputs.append(next_content)
    716         nextparameters.append(x._parameters)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:373, in broadcast_to_offsets_avoiding_carry(list_content, offsets)
    371         return list_content.content[:next_length]
    372     else:
--> 373         return list_content._broadcast_tooffsets64(offsets).content
    374 elif isinstance(list_content, ListArray):
    375     # Is this list contiguous?
    376     if nplike.array_equal(
    377         list_content.starts.data[1:], list_content.stops.data[:-1]
    378     ):
    379         # Does this list match the offsets?

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listoffsetarray.py:439, in ListOffsetArray._broadcast_tooffsets64(self, offsets)
    434     next_content = self._content[this_start:]
    436 if nplike.known_data and not nplike.array_equal(
    437     this_zero_offsets, offsets.data
    438 ):
--> 439     raise ValueError("cannot broadcast nested list")
    441 return ListOffsetArray(
    442     offsets, next_content[: offsets[-1]], parameters=self._parameters
    443 )

ValueError: cannot broadcast nested list

This error occurred while calling

    numpy.add.__call__(
        <Array [[], [], ..., [25], [36, 49, 64, 81]] type='5 * var * int64'>
        <Array [[], [], ..., [43.6, 59.3, 77.4, 98]] type='5 * var * float64'>
    )
```

Sometimes, these misalignments are overt, but sometimes they’re subtle and embedded deep within a very large array. You can start investigating a problem like this with [`ak.num()`](sphinx-llm:137e3487aaa0453fa61d9c86fc644b7b#ak.num):

```ipython3
ak.num(sliced_x) != ak.num(sliced_y)
```

```ipython3
np.nonzero(ak.to_numpy(ak.num(sliced_x) != ak.num(sliced_y)))
```

But it’s also possible to avoid them in the first place.

## Masking with missing values

The problem was that the two arrays’ shapes changed differently; instead, we’ll slice them in such a way that their shapes don’t change at all.

The [`ak.mask()`](sphinx-llm:9a507fdd6567423185d9261d0dc62bad#ak.mask) function uses a boolean array like a slice, but takes values that line up with `False` and returns `None` instead of removing them.

```ipython3
ak.mask(array_x, array_x > 3)
```

It can also be accessed as an array property, with square brackets, so that it resembles a slice:

```ipython3
masked_x = array_x.mask[array_x > 3]
masked_y = array_y.mask[array_y > 3]
```

```ipython3
masked_x
```

```ipython3
masked_y
```

The results of these two masks can be used in a mathematical expression because they line up:

```ipython3
result = masked_x**2 + masked_y**2
result
```

Now only one problem remains: the `None` (missing) values might be undesirable in the output. There are several ways to get rid of them:

* [`ak.drop_none()`](sphinx-llm:2495120d552d4c9f9d081cd38e6b71b4#ak.drop_none) eliminates `None`, like a slice, but it can be done once at the end of a calculation,
* [`ak.fill_none()`](sphinx-llm:e26b7c2675674008bc03fc5f18da53c5#ak.fill_none) replaces `None` with a chosen value,
* [`ak.flatten()`](sphinx-llm:6b8f65e7187745b09f39a6bdf1966b16#ak.flatten) removes list structure, and if the `None` values are at the level of a list (the ones in `result` aren’t), they’ll be removed too,
* [`ak.singletons()`](sphinx-llm:744acdbcea7349c7b3acb92b357d5d28#ak.singletons) replaces `None` with `[]` and any other value `x` with `[x]`. The resulting lists all have length 0 or length 1.

```ipython3
ak.drop_none(result, axis=1)
```

```ipython3
ak.fill_none(result, -1, axis=1)
```

```ipython3
ak.singletons(result, axis=1)
```

As a final note, the difference between using [`ak.drop_none()`](sphinx-llm:2495120d552d4c9f9d081cd38e6b71b4#ak.drop_none) and slicing with the result of [`ak.is_none()`](sphinx-llm:488667031383491ba3420edcd0c7126a#ak.is_none) is that [`ak.drop_none()`](sphinx-llm:2495120d552d4c9f9d081cd38e6b71b4#ak.drop_none) also removes “missingness” from the data type; a slice does not.

```ipython3
result[~ak.is_none(result, axis=1)]
```

(Note the `?` for “option-type” before `float64`. This could have consequences, good or bad, at a later stage in processing.)
