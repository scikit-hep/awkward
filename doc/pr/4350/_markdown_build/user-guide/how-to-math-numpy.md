# How to perform computations with NumPy

Awkward Array’s integration with NumPy allows you to use NumPy’s array functions on data with complex structures, including ragged and heterogeneous arrays.

```ipython3
import awkward as ak
import numpy as np
```

## Universal functions (ufuncs)

[NumPy’s universal functions (ufuncs)](https://numpy.org/doc/stable/reference/ufuncs.html) are functions that operate elementwise on arrays. They are broadcasting-aware, so they can naturally handle data structures like ragged arrays that are common in Awkward Arrays.

Here’s an example of applying `np.sqrt`, a NumPy ufunc, to an Awkward Array:

```ipython3
data = ak.Array([[1, 4, 9], [], [16, 25]])

np.sqrt(data)
```

Notice that the ufunc applies to the numeric data, passing through all dimensions of nested lists, even if those lists have variable length. This also applies to heterogeneous data, in which the data are not all of the same type.

```ipython3
data = ak.Array([[1, 4, 9], [], 16, [[[25]]]])

np.sqrt(data)
```

Unary and binary operations on Awkward Arrays, such as `+`, `-`, `>`, and `==`, are actually calling NumPy ufuncs. For instance, `+`:

```ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([[10, 20, 30], [], [40, 50]])

array1 + array2
```

is actually `np.add`:

```ipython3
np.add(array1, array2)
```

### Arrays with record fields

Ufuncs can only be applied to numerical data in lists, not records.

```ipython3
records = ak.Array([{"x": 4, "y": 9}, {"x": 16, "y": 25}])
```

```ipython3
np.sqrt(records)
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[7], line 1
----> 1 np.sqrt(records)

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1174, in apply_step.<locals>.continuation()
   1172 # Any RecordArrays?
   1173 elif any(x.is_record for x in contents):
-> 1174     return broadcast_any_record()
   1176 else:
   1177     raise ValueError(
   1178         "cannot broadcast: {}{}".format(
   1179             ", ".join(repr(type(x)) for x in inputs), in_function(options)
   1180         )
   1181     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:490, in apply_step.<locals>.broadcast_any_record()
    488 def broadcast_any_record():
    489     if not options["allow_records"]:
--> 490         raise ValueError(f"cannot broadcast records{in_function(options)}")
    492     frozen_record_fields: frozenset[str] | None = UNSET
    493     first_record = next(c for c in contents if c.is_record)

ValueError: cannot broadcast records in sqrt

This error occurred while calling

    numpy.sqrt.__call__(
        <Array [{x: 4, y: 9}, {x: 16, ...}] type='2 * {x: int64, y: int64}'>
    )
```

However, you can pull each field out of a record and apply the ufunc to it.

```ipython3
np.sqrt(records.x)
```

```ipython3
np.sqrt(records.y)
```

If you want the result wrapped up in a new array of records, you can use [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) to do that.

```ipython3
ak.zip({"x": np.sqrt(records.x), "y": np.sqrt(records.y)})
```

Here’s an idiom that would apply a ufunc to every field individually, and then wrap up the result as a new record with the same fields (using [`ak.fields()`](sphinx-llm:056b2c11871845eaa8962e15bfbffd9e#ak.fields), [`ak.unzip()`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip), and [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip)):

```ipython3
ak.zip({key: np.sqrt(value) for key, value in zip(ak.fields(records), ak.unzip(records))})
```

The reaons that Awkward Array does not do this automatically is to prevent mistakes: it’s common for records to represent coordinates of data points, and if the coordinates are not Cartesian, the one-to-one application is not correct.

### Using non-NumPy ufuncs

NumPy-compatible ufuncs exist in other libraries, like SciPy, and can be applied in the same way. Here’s how you can apply `scipy.special.gamma` and `scipy.special.erf`:

```ipython3
import scipy.special

data = ak.Array([[0.1, 0.2, 0.3], [], [0.4, 0.5]])
```

```ipython3
scipy.special.gamma(data)
```

```ipython3
scipy.special.erf(data)
```

You can even create your own ufuncs using Numba’s `@nb.vectorize`:

```ipython3
import numba as nb

@nb.vectorize
def gcd_euclid(x, y):
    # computation that is more complex than a formula
    while y != 0:
        x, y = y, x % y
    return x
```

```ipython3
x = ak.Array([[10, 20, 30], [], [40, 50]])
y = ak.Array([[5, 40, 15], [], [24, 255]])
```

```ipython3
gcd_euclid(x, y)
```

Since Numba has JIT-compiled this function, it would run much faster on large arrays than custom Python code.

## Non-ufunc NumPy functions

Some NumPy functions don’t satisfy the ufunc protocol, but have been implemented for Awkward Arrays because they are useful. You can tell when a NumPy function has an Awkward Array implementation when a function with the same name and signature exists in both libraries.

For instance, `np.where` works on Awkward Arrays because [`ak.where()`](sphinx-llm:60ec4530e4274bec8aa8df6ce404979e#ak.where) exists:

```ipython3
np.where(y % 2 == 0, x, y) 
```

(The above selects elements from `x` when `y` is even and elements from `y` when `y` is odd.)

Similarly, `np.concatenate` works on Awkward Arrays because [`ak.concatenate()`](sphinx-llm:9d688ff77559406881cf7aa931110693#ak.concatenate) exists:

```ipython3
np.concatenate([x, y])
```

```ipython3
np.concatenate([x, y], axis=1)
```

Other NumPy functions, without an equivalent in the Awkward Array library, will work only if the Awkward Array can be converted into a NumPy array.

Ragged arrays can’t be converted to NumPy:

```ipython3
np.fft.fft(ak.Array([[1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]))
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[21], line 1
----> 1 np.fft.fft(ak.Array([[1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]))

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1661, in Array.__array_function__(self, func, types, args, kwargs)
   1647 def __array_function__(self, func, types, args, kwargs):
   1648     """
   1649     Intercepts attempts to pass this Array to those NumPy functions other
   1650     than universal functions that have an Awkward equivalent.
   (...)   1659     See also #__array_ufunc__.
   1660     """
-> 1661     return ak._connect.numpy.array_function(
   1662         func, types, args, kwargs, behavior=self._behavior, attrs=self._attrs
   1663     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:109, in array_function(func, types, args, kwargs, behavior, attrs)
    106 unique_backends = frozenset(_find_backends(all_arguments))
    107 backend = common_backend(unique_backends)
--> 109 rectilinear_args = tuple(_to_rectilinear(x, backend) for x in args)
    110 rectilinear_kwargs = {k: _to_rectilinear(v, backend) for k, v in kwargs.items()}
    111 result = func(*rectilinear_args, **rectilinear_kwargs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:109, in <genexpr>(.0)
    106 unique_backends = frozenset(_find_backends(all_arguments))
    107 backend = common_backend(unique_backends)
--> 109 rectilinear_args = tuple(_to_rectilinear(x, backend) for x in args)
    110 rectilinear_kwargs = {k: _to_rectilinear(v, backend) for k, v in kwargs.items()}
    111 result = func(*rectilinear_args, **rectilinear_kwargs)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:78, in _to_rectilinear(arg, backend)
     69     # Otherwise, cast to layout and convert
     70     else:
     71         layout = ak.to_layout(
     72             arg,
     73             allow_record=False,
   (...)     76             string_policy="error",
     77         )
---> 78         return layout.to_backend(backend).to_backend_array(allow_missing=True)
     79 elif isinstance(arg, tuple):
     80     return tuple(_to_rectilinear(x, backend) for x in arg)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:1131, in Content.to_backend_array(self, allow_missing, backend)
   1129 else:
   1130     backend = regularize_backend(backend)
-> 1131 return self._to_backend_array(allow_missing, backend)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listoffsetarray.py:1900, in ListOffsetArray._to_backend_array(self, allow_missing, backend)
   1898     return buffer.view(np.dtype(("S", max_count)))
   1899 else:
-> 1900     return self.to_RegularArray()._to_backend_array(allow_missing, backend)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listoffsetarray.py:296, in ListOffsetArray.to_RegularArray(self)
    291 _size = Index64.empty(1, self._backend.nplike)
    292 assert (
    293     _size.nplike is self._backend.nplike
    294     and self._offsets.nplike is self._backend.nplike
    295 )
--> 296 self._backend.maybe_kernel_error(
    297     self._backend[
    298         "awkward_ListOffsetArray_toRegularArray",
    299         _size.dtype.type,
    300         self._offsets.dtype.type,
    301     ](
    302         _size.data,
    303         self._offsets.data,
    304         self._offsets.length,
    305     )
    306 )
    307 size = self._backend.nplike.index_as_shape_item(_size[0])
    308 length = self._offsets.length - 1

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_backends/backend.py:62, in Backend.maybe_kernel_error(self, error)
     60     return
     61 else:
---> 62     raise ValueError(self.format_kernel_error(error))

ValueError: cannot convert to RegularArray because subarray lengths are not regular (in compiled code: https://github.com/scikit-hep/awkward/blob/awkward-cpp-56/awkward-cpp/src/cpu-kernels/awkward_ListOffsetArray_toRegularArray.cpp#L22)
```

But arrays with equal-sized lists can:

```ipython3
np.fft.fft(ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
```
