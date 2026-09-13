# How Awkward broadcasting works

Functions that accept more than one array argument need to combine the elements of their array elements somehow, particularly if the input arrays have different numbers of dimensions. That combination is called “broadcasting.” Broadcasting in Awkward Array is very similar to [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), with some minor differences described at the end of this section.

```ipython3
import awkward as ak
import numpy as np
```

## Broadcasting in mathematical functions

Any function that takes more than one array argument has to broadcast them together; a common case of that is in binary operators of a mathematical expression:

```ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

The single `10` in `array2` is added to every element of `[1, 2, 3]` in `array1`, and the single `30` is added to every element of `[4, 5]`. The single `20` in `array2` is not added to anything in `array1` because the corresponding list is empty.

For broadcasting to be successful, the arrays need to have the same length in all dimensions except the one being broadcasted; `array1` and `array2` both had to be length 3 in the example above. That’s why this example fails:

```ipython3
array1 = ak.Array([[1, 2, 3], [4, 5]])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[3], line 4
      1 array1 = ak.Array([[1, 2, 3], [4, 5]])
      2 array2 = ak.Array([10, 20, 30])
      3 
----> 4 array1 + array2

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:655, in apply_step.<locals>.broadcast_any_list()
    653         nextparameters.append(x._parameters)
    654     else:
--> 655         raise ValueError(
    656             "cannot broadcast RegularArray of size "
    657             f"{x.size} with RegularArray of size {dim_size}{in_function(options)}"
    658         )
    659 else:
    660     nextinputs.append(x)

ValueError: cannot broadcast RegularArray of size 2 with RegularArray of size 3 in add

This error occurred while calling

    numpy.add.__call__(
        <Array [[1, 2, 3], [4, 5]] type='2 * var * int64'>
        <Array [10, 20, 30] type='3 * int64'>
    )
```

The same applies to functions of multiple arguments that aren’t associated with any binary operator:

```ipython3
array1 = ak.Array([[True, False, True], [], [False, True]])
array2 = ak.Array([True, True, False])

np.logical_and(array1, array2)
```

And functions that aren’t universal functions (ufuncs):

```ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([10, 20, 30])

np.where(array1 % 2 == 0, array1, array2)
```

## Using `ak.broadcast_arrays`

Sometimes, you may want to broadcast arrays to a common shape without performing an additional operation. The [`ak.broadcast_arrays()`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays) function allows you to do this:

```ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([10, 20, 30])

ak.broadcast_arrays(array1, array2)
```

This code would align `array1` and `array2` into compatible shapes that can be used in subsequent operations, effectively showing how each element corresponds between the two original arrays.

## Missing data, heterogeneous data, and records

One of the ways Awkward Arrays extend beyond NumPy is by allowing the use of `None` for missing data. These `None` values are broadcasted like empty lists:

```ipython3
array1 = ak.Array([[1, 2, 3], None, [4, 5]])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

Another difference from NumPy is that Awkward Arrays can contain data of mixed type, such as different numbers of dimensions. If numerical values *can* be matched across such arrays, they are:

```ipython3
array1 = ak.Array([[1, 2, 3], 4, 5])
array2 = ak.Array([10, 20, 30])

array1 + array2
```

Arrays containing records can also be broadcasted, though most mathematical operations cannot be applied to records. Here is an example using [`ak.broadcast_arrays()`](sphinx-llm:ecf144fb88044843b0e932a031de236c#ak.broadcast_arrays).

```ipython3
array1 = ak.Array([
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
])
array2 = ak.Array([10, 20, 30])

ak.broadcast_arrays(array1, array2)
```

## Differences from NumPy broadcasting

Awkward Array broadcasting is identical to NumPy broadcasting in three respects:

1. arrays with the same number of dimensions must match lengths (except for length 1) exactly,
2. length-1 dimensions expand like scalars (one to many),
3. for arrays with different numbers of dimensions, the smaller number of dimensions is expanded to match the largest number of dimensions.

Awkward Arrays with fixed-length dimensions—not “variable-length” or “ragged”—broadcast exactly like NumPy.

Awkward Arrays with ragged dimensions expand the smaller number of dimensions on the left, whereas NumPy and Awkward-with-fixed-length expand the smaller number of dimensions on the right, when implementing point 3 above. This is the only difference.

Here’s a demonstration of NumPy broadcasting:

```ipython3
x = np.array([
    [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
])
y = np.array([
    [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
    [[100, 200, 300, 400], [500, 600, 700, 800], [900, 1000, 1100, 1200]],
])

x + y
```

And fixed-length Awkward Arrays made from these can be broadcasted the same way:

```ipython3
ak.Array(x) + ak.Array(y)
```

but only because the latter have completely regular dimensions, like their NumPy counterparts.

```ipython3
print(x.shape)
print(y.shape)
```

```myst-ansi
(3, 4)
(2, 3, 4)
```

```ipython3
print(ak.Array(x).type)
print(ak.Array(y).type)
```

```myst-ansi
3 * 4 * int64
2 * 3 * 4 * int64
```

In both NumPy and Awkward Array, `x` has fewer dimensions than `y`, so `x` is expanded on the *left* from length-1 to length-2.

However, if the Awkward Array has variable-length *type*, regardless of whether the actual lists have variable lengths,

```ipython3
print(ak.Array(x.tolist()).type)
print(ak.Array(y.tolist()).type)
```

```myst-ansi
3 * var * int64
2 * var * var * int64
```

this broadcasting does not work:

```ipython3
ak.Array(x.tolist()) + ak.Array(y.tolist())
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[15], line 1
----> 1 ak.Array(x.tolist()) + ak.Array(y.tolist())

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:655, in apply_step.<locals>.broadcast_any_list()
    653         nextparameters.append(x._parameters)
    654     else:
--> 655         raise ValueError(
    656             "cannot broadcast RegularArray of size "
    657             f"{x.size} with RegularArray of size {dim_size}{in_function(options)}"
    658         )
    659 else:
    660     nextinputs.append(x)

ValueError: cannot broadcast RegularArray of size 2 with RegularArray of size 3 in add

This error occurred while calling

    numpy.add.__call__(
        <Array [[1, 2, 3, 4], ..., [9, 10, 11, 12]] type='3 * var * int64'>
        <Array [[[10, 20, 30, 40], ...], ...] type='2 * var * var * int64'>
    )
```

Instead of trying to add a dimension to the left of `x`’s shape, `(3, 4)`, to make `(2, 3, 4)`, the ragged broadcasting is trying to add a dimension to the right of `x`’s shape, and it doesn’t line up.

### Why does ragged broadcasting have to be different?

Instead of adding a new dimension on the left, as NumPy and fixed-length Awkward Arrays do, ragged broadcasting tries to add a new dimension on the right in order to make it useful for emulating imperative code like

```ipython3
for x_i, y_i in zip(x, y):
    for x_ij, y_ij in zip(x_i, y_i):
        print("[", end=" ")
        for y_ijk in y_ij:
            print(x_ij + y_ijk, end=" ")
        print("]")
    print()
```

```myst-ansi
[ 11 21 31 41 ]
[ 52 62 72 82 ]
[ 93 103 113 123 ]

[ 105 205 305 405 ]
[ 506 606 706 806 ]
[ 907 1007 1107 1207 ]

```

In the above, the value of `x_ij` is not varying while `y_ijk` varies in the innermost for-loop. In imperative code like this, it’s natural for the outermost (left-most) dimensions of two nested lists to line up, while a scalar from the list with fewer dimensions, `x`, stays constant (is effectively duplicated) for each innermost `y` value.

This is *not* what NumPy’s left-broadcasting does:

```ipython3
x + y
```

Notice that the numerical values are different!

To get the behavior we expect from imperative code, we need to right-broadcast, which is what ragged broadcasting in Awkward Array does:

```ipython3
x = ak.Array([
    [1.1, 2.2, 3.3],
    [],
    [4.4, 5.5]
])
y = ak.Array([
    [[1], [1, 2], [1, 2, 3]],
    [],
    [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
])

for x_i, y_i in zip(x, y):
    print("[")
    for x_ij, y_ij in zip(x_i, y_i):
        print("    [", end=" ")
        for y_ijk in y_ij:
            print(x_ij + y_ijk, end=" ")
        print("]")
    print("]\n")

x + y
```

```myst-ansi
[
    [ 2.1 ]
    [ 3.2 4.2 ]
    [ 4.3 5.3 6.3 ]
]

[
]

[
    [ 5.4 6.4 7.4 8.4 ]
    [ 6.5 7.5 8.5 9.5 10.5 ]
]

```

In summary,

* NumPy left-broadcasts,
* Awkward Arrays with fixed-length lists left-broadcast, for consistency with NumPy,
* Awkward Arrays with variable-length lists right-broadcast, for consistency with imperative code.

One way to control this is to ensure that all arrays involved in an expression have the same number of dimensions by explicitly expanding them. Implicit broadcasting only happens for arrays of different numbers of dimensions, or if the length of a dimension is 1.

But it might also be the case that your arrays have lists of equal length, so they seem to be regular like a NumPy array, yet their data type says that the lists *can* be variable-length. Perhaps you got the NumPy-like data from a source that doesn’t enforce fixed lengths, such as Python lists ([`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter)), JSON ([`ak.from_json()`](sphinx-llm:8dc434abe56445a0ba5c48a99f18b65e#ak.from_json)), or Parquet ([`ak.from_parquet()`](sphinx-llm:ab272f6edef4436f8edba845c0477c5f#ak.from_parquet)). Check the array’s [`ak.type()`](sphinx-llm:65d3dedfc8ad48c98f3bacc6472ae8c8#ak.type) to see whether all dimensions are ragged (`var *`) or regular (some number `*`).

The [`ak.from_regular()`](sphinx-llm:5977f2bff2984a609f29aa5c2f0eef19#ak.from_regular) and [`ak.to_regular()`](sphinx-llm:2a2dc6754b2948d79f20c53eb2b3d2b9#ak.to_regular) functions toggle ragged (`var *`) and regular (some number `*`) dimensions, and [`ak.enforce_type()`](sphinx-llm:ba952268b279401086f48d422c6fd7b9#ak.enforce_type) can be used to cast types like this in general.
