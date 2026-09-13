# How to create arrays of strings

Awkward Arrays can contain strings, although these strings are just a special view of lists of `uint8` numbers. As such, the variable-length data are efficiently stored.

NumPy’s strings are padded to have equal width, and Pandas’s strings are Python objects. Awkward Array doesn’t have nearly as many functions for manipulating arrays of strings as NumPy and Pandas, though.

```ipython3
import awkward as ak
import numpy as np
```

## From Python strings

The [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) constructor and [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) recognize strings, and strings are returned by [`ak.to_list()`](sphinx-llm:c5d0843c37c34cb88bbec16becc6a6f0#ak.to_list).

```ipython3
ak.Array(["one", "two", "three"])
```

They may be nested within anything.

```ipython3
ak.Array([["one", "two"], [], ["three"]])
```

## From NumPy arrays

NumPy strings are also recognized by [`ak.from_numpy()`](sphinx-llm:efd7416013fa442595132125c93aa4ea#ak.from_numpy) and [`ak.to_numpy()`](sphinx-llm:0dc12c9b410c4566b8e6578a4b063dbf#ak.to_numpy).

```ipython3
numpy_array = np.array(["one", "two", "three", "four"])
numpy_array
```

```ipython3
awkward_array = ak.Array(numpy_array)
awkward_array
```

## Operations with strings

Since strings are really just lists, some of the list operations “just work” on strings.

```ipython3
ak.num(awkward_array)
```

```ipython3
awkward_array[:, 1:]
```

Others had to be specially overloaded for the string case, such as string-equality. The default meaning for `==` would be to descend to the lowest level and compare numbers (characters, in this case).

```ipython3
awkward_array == "three"
```

```ipython3
awkward_array == ak.Array(["ONE", "TWO", "three", "four"])
```

Similarly, [`ak.sort()`](sphinx-llm:04e24ad5c9624efeab7a13c95c4f71cf#ak.sort) and [`ak.argsort()`](sphinx-llm:1ac2c8dad5854d5eb0608d90863ee4e6#ak.argsort) sort strings lexicographically, not individual characters.

```ipython3
ak.sort(awkward_array)
```

Still other operations had to be inhibited, since they wouldn’t make sense for strings.

```ipython3
np.sqrt(awkward_array)
```

```ipythontb
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[11], line 1
----> 1 np.sqrt(awkward_array)

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_broadcasting.py:1183, in apply_step(backend, inputs, action, depth, depth_context, lateral_context, options)
   1176     else:
   1177         raise ValueError(
   1178             "cannot broadcast: {}{}".format(
   1179                 ", ".join(repr(type(x)) for x in inputs), in_function(options)
   1180             )
   1181         )
-> 1183 result = action(
   1184     inputs,
   1185     depth=depth,
   1186     depth_context=depth_context,
   1187     lateral_context=lateral_context,
   1188     continuation=continuation,
   1189     backend=backend,
   1190     options=options,
   1191 )
   1193 if isinstance(result, tuple) and all(isinstance(x, Content) for x in result):
   1194     if any(content.backend is not backend for content in result):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_connect/numpy.py:420, in array_ufunc.<locals>.action(inputs, **ignore)
    415     # Do we have all-strings? If so, we can't proceed
    416     if all(
    417         x.is_list and x.parameter("__array__") in ("string", "bytestring")
    418         for x in contents
    419     ):
--> 420         raise TypeError(
    421             f"{type(ufunc).__module__}.{ufunc.__name__} is not implemented for string types. "
    422             "To register an implementation, add a name to these string(s) and register a behavior overload"
    423         )
    425 if ufunc is numpy.matmul:
    426     raise NotImplementedError(
    427         "matrix multiplication (`@` or `np.matmul`) is not yet implemented for Awkward Arrays"
    428     )

TypeError: numpy.sqrt is not implemented for string types. To register an implementation, add a name to these string(s) and register a behavior overload

This error occurred while calling

    numpy.sqrt.__call__(
        <Array ['one', 'two', 'three', 'four'] type='4 * string'>
    )
```

## Categorical strings

A large set of strings with few unique values are more efficiently manipulated as integers than as strings. In Pandas, this is [categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html), in R, it’s called a [factor](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/factor), and in Arrow and Parquet, it’s [dictionary encoding](https://arrow.apache.org/blog/2019/09/05/faster-strings-cpp-parquet/).

The [`ak.str.to_categorical()`](sphinx-llm:ad92a02f22f4473ebb80165887f3828c#ak.str.to_categorical) (requires PyArrow) function makes Awkward Arrays categorical in this sense. [`ak.to_arrow()`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow) and [`ak.to_parquet()`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet) recognize categorical data and convert it to the corresponding Arrow and Parquet types.

```ipython3
uncategorized = ak.Array(["three", "one", "two", "two", "three", "one", "one", "one"])
uncategorized
```

```ipython3
categorized = ak.str.to_categorical(uncategorized)
categorized
```

Internally, the data now have an index that selects from a set of unique strings.

```ipython3
categorized.layout.index
```

```ipython3
ak.Array(categorized.layout.content)
```

The main advantage to Awkward categorical data (other than proper conversions to Arrow and Parquet) is that equality is performed using the index integers.

```ipython3
categorized == "one"
```

## With ArrayBuilder

[`ak.ArrayBuilder()`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) is described in more detail [in this tutorial](sphinx-llm:26cb6d67e3ae4372aacb2b87a0852a58), but you can add strings by calling the `string` method or simply appending them.

(This is what [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) uses internally to accumulate data.)

```ipython3
builder = ak.ArrayBuilder()

builder.string("one")
builder.append("two")
builder.append("three")

array = builder.snapshot()
array
```
