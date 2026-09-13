# How to create arrays of missing data

Data at any level of an Awkward Array can be “missing,” represented by `None` in Python.

This functionality is somewhat like NumPy’s [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html), but masked arrays can only declare numerical values to be missing (not, for instance, a row of a 2-dimensional array) and they represent missing data with an `np.ma.masked` object instead of `None`.

Pandas also handles missing data, but in several different ways. For floating point columns, `NaN` (not a number) is used to mean “missing,” and [as of version 1.0](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#missing-data-na), Pandas has a `pd.NA` object for missing data in other data types.

In Awkward Array, floating point `NaN` and a missing value are clearly distinct. Missing data, like all data in Awkward Arrays, are also not represented by any Python object; they are converted *to* and *from* `None` by [`ak.to_list()`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) and [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter).

```ipython3
import awkward as ak
import numpy as np
```

## From Python None

The [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) constructor and [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) interpret `None` as a missing value, and [`ak.to_list()`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list) converts them back into `None`.

```ipython3
ak.Array([1, 2, 3, None, 4, 5])
```

The missing values can be deeply nested (missing integers):

```ipython3
ak.Array([[[[], [1, 2, None]]], [[[3]]], []])
```

They can be shallow (missing lists):

```ipython3
ak.Array([[[[], [1, 2]]], None, [[[3]]], []])
```

Or both:

```ipython3
ak.Array([[[[], [3]]], None, [[[None]]], []])
```

Records can also be missing:

```ipython3
ak.Array([{"x": 1, "y": 1}, None, {"x": 2, "y": 2}])
```

Potentially missing values are represented in the type string as “`?`” or “`option[...]`” (if the nested type is a list, which needs to be bracketed for clarity).

## From NumPy arrays

Normal NumPy arrays can’t represent missing data, but masked arrays can. Here is how one is constructed in NumPy:

```ipython3
numpy_array = np.ma.MaskedArray([1, 2, 3, 4, 5], [False, False, True, True, False])
numpy_array
```

It returns `np.ma.masked` objects if you try to access missing values:

```ipython3
numpy_array[0], numpy_array[1], numpy_array[2], numpy_array[3], numpy_array[4]
```

But it uses `None` for missing values in `tolist`:

```ipython3
numpy_array.tolist()
```

The [`ak.from_numpy()`](sphinx-llm:efd7416013fa442595132125c93aa4ea#ak.from_numpy) function converts masked arrays into Awkward Arrays with missing values, as does the [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) constructor.

```ipython3
awkward_array = ak.Array(numpy_array)
awkward_array
```

The reverse, [`ak.to_numpy()`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy), returns masked arrays if the Awkward Array has missing data.

```ipython3
ak.to_numpy(awkward_array)
```

But [np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html), the usual way of casting data as NumPy arrays, does not. ([np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html) is supposed to return a plain [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), which [np.ma.masked_array](https://numpy.org/doc/stable/reference/generated/numpy.ma.masked_array.html) is not.)

```ipython3
np.asarray(awkward_array)
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[12], line 1
----> 1 np.asarray(awkward_array)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1563, in Array.__array__(self, dtype, copy)
   1534 def __array__(self, dtype=None, copy=None):
   1535     """
   1536     Intercepts attempts to convert this Array into a NumPy array and
   1537     either performs a conversion if possible or raises an error.
   (...)   1561     cannot be sliced as dimensions.
   1562     """
-> 1563     with ak._errors.OperationErrorContext(
   1564         "numpy.asarray", (self,), {"dtype": dtype, "copy": copy}
   1565     ):
   1566         from awkward._connect.numpy import convert_to_array
   1568         return convert_to_array(self._layout, dtype=dtype, copy=copy)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1568, in Array.__array__(self, dtype, copy)
   1563 with ak._errors.OperationErrorContext(
   1564     "numpy.asarray", (self,), {"dtype": dtype, "copy": copy}
   1565 ):
   1566     from awkward._connect.numpy import convert_to_array
-> 1568     return convert_to_array(self._layout, dtype=dtype, copy=copy)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:526, in convert_to_array(layout, dtype, copy)
    525 def convert_to_array(layout, dtype=None, copy=None):
--> 526     out = ak.operations.to_numpy(layout, allow_missing=False)
    527     if copy:
    528         return numpy.array(out, dtype=dtype, copy=True)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_dispatch.py:66, in named_high_level_function.<locals>.dispatch(*args, **kwargs)
     64 # Failed to find a custom overload, so resume the original function
     65 try:
---> 66     next(gen_or_result)
     67 except StopIteration as err:
     68     return err.value

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_to_numpy.py:48, in to_numpy(array, allow_missing)
     45 yield (array,)
     47 # Implementation
---> 48 return _impl(array, allow_missing)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_to_numpy.py:60, in _impl(array, allow_missing)
     57 backend = NumpyBackend.instance()
     58 numpy_layout = layout.to_backend(backend)
---> 60 return numpy_layout.to_backend_array(allow_missing=allow_missing)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:1131, in Content.to_backend_array(self, allow_missing, backend)
   1129 else:
   1130     backend = regularize_backend(backend)
-> 1131 return self._to_backend_array(allow_missing, backend)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/bytemaskedarray.py:1088, in ByteMaskedArray._to_backend_array(self, allow_missing, backend)
   1087 def _to_backend_array(self, allow_missing, backend):
-> 1088     return self.to_IndexedOptionArray64()._to_backend_array(allow_missing, backend)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/indexedoptionarray.py:1617, in IndexedOptionArray._to_backend_array(self, allow_missing, backend)
   1615         return nplike.ma.MaskedArray(data, mask)
   1616     else:
-> 1617         raise ValueError(
   1618             "Content.to_nplike cannot convert 'None' values to "
   1619             "np.ma.MaskedArray unless the "
   1620             "'allow_missing' parameter is set to True"
   1621         )
   1622 else:
   1623     if allow_missing:

ValueError: Content.to_nplike cannot convert 'None' values to np.ma.MaskedArray unless the 'allow_missing' parameter is set to True

This error occurred while calling

    numpy.asarray(
        <Array [1, 2, None, None, 5] type='5 * ?int64'>
        dtype = None
        copy = None
    )
```

## Missing rows vs missing numbers

In Awkward Array, a missing list is a different thing from a list whose values are missing. However, [`ak.to_numpy()`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy) converts it for you.

```ipython3
missing_row = ak.Array([[1, 2, 3], None, [4, 5, 6]])
missing_row
```

```ipython3
ak.to_numpy(missing_row)
```

## NaN is not missing

Floating point `NaN` values are simply unrelated to missing values, in both Awkward Array and NumPy.

```ipython3
missing_with_nan = ak.Array([1.1, 2.2, np.nan, None, 3.3])
missing_with_nan
```

```ipython3
ak.to_numpy(missing_with_nan)
```

## Missing values as empty lists

Sometimes, it’s useful to think about a potentially missing value as a length-1 list if it is not missing and a length-0 list if it is. (Some languages define the [option type as a kind of list](https://www.scala-lang.org/api/2.13.3/scala/Option.html).)

The Awkward functions [`ak.singletons()`](sphinx-llm:744acdbcea7349c7b3acb92b357d5d28#ak.singletons) and [`ak.firsts()`](sphinx-llm:b6d811d080f84b19a2cb464c8aa1a1e4#ak.firsts) convert from “`None` form” to and from “lists form.”

```ipython3
none_form = ak.Array([1, 2, 3, None, None, 5])
none_form
```

```ipython3
lists_form = ak.singletons(none_form)
lists_form
```

```ipython3
ak.firsts(lists_form)
```

## Masking instead of slicing

The most common way of filtering data is to slice it with an array of booleans (usually the result of a calculation).

```ipython3
array = ak.Array([1, 2, 3, 4, 5])
array
```

```ipython3
booleans = ak.Array([True, True, False, False, True])
booleans
```

```ipython3
array[booleans]
```

The data can also be effectively filtered by replacing values with `None`. The following syntax does that:

```ipython3
array.mask[booleans]
```

(Or use the [`ak.mask()`](sphinx-llm:9a507fdd6567423185d9261d0dc62bad#ak.mask) function.)

An advantage of masking is that the length and nesting structure of the masked array is the same as the original array, so anything that broadcasts with one broadcasts with the other (so that unfiltered data can be used interchangeably with filtered data).

```ipython3
array + array.mask[booleans]
```

whereas

```ipython3
array + array[booleans]
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[25], line 1
----> 1 array + array[booleans]

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

ValueError: cannot broadcast RegularArray of size 3 with RegularArray of size 5 in add

This error occurred while calling

    numpy.add.__call__(
        <Array [1, 2, 3, 4, 5] type='5 * int64'>
        <Array [1, 2, 5] type='3 * int64'>
    )
```

## With ArrayBuilder

[`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) is described in more detail [in this tutorial](sphinx-llm:26cb6d67e3ae4372aacb2b87a0852a58), but you can add missing values to an array using the `null` method or appending `None`.

(This is what [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) uses internally to accumulate data.)

```ipython3
builder = ak.ArrayBuilder()

builder.append(1)
builder.append(2)
builder.null()
builder.append(None)
builder.append(3)

array = builder.snapshot()
array
```

## In Numba

Functions that Numba Just-In-Time (JIT) compiles can use [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) or construct a boolean array for [`ak.mask()`](sphinx-llm:9a507fdd6567423185d9261d0dc62bad#ak.mask).

([`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) can’t be constructed or converted to an array using `snapshot` inside a JIT-compiled function, but can be outside the compiled context.)

```ipython3
import numba as nb
```

```ipython3
@nb.jit
def example(builder):
    builder.append(1)
    builder.append(2)
    builder.null()
    builder.append(None)
    builder.append(3)
    return builder


builder = example(ak.ArrayBuilder())

array = builder.snapshot()
array
```

```ipython3
@nb.jit
def faster_example():
    data = np.empty(5, np.int64)
    mask = np.empty(5, np.bool_)
    data[0] = 1
    mask[0] = True
    data[1] = 2
    mask[1] = True
    mask[2] = False
    mask[3] = False
    data[4] = 5
    mask[4] = True
    return data, mask


data, mask = faster_example()

array = ak.mask(data, mask)
array
```
