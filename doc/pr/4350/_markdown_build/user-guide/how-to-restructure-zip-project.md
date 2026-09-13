# How to restructure arrays with zip/unzip and project

```ipython3
%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"
```

## Unzipping an array of records

As discussed in [How to create arrays of records](sphinx-llm:e2a08ad149184f49b0882e363831b487), in addition to primitive types like [`numpy.float64`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float64) and [`numpy.datetime64`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64), Awkward Arrays can also contain records. These records are formed from a fixed number of optionally named *fields*.

```ipython3
import awkward as ak
import numpy as np

records = ak.Array(
    [
        {"x": 1, "y": 1.1, "z": "one"},
        {"x": 2, "y": 2.2, "z": "two"},
        {"x": 3, "y": 3.3, "z": "three"},
        {"x": 4, "y": 4.4, "z": "four"},
        {"x": 5, "y": 5.5, "z": "five"},
    ]
)
```

Although it is useful to be able to create arrays from a sequence of records (as [arrays of structures](https://en.wikipedia.org/wiki/AoS_and_SoA#Array_of_structures)), Awkward Array implements arrays as [*structures of arrays*](https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays). It is therefore more natural to think about arrays in terms of their fields.
In the above example, we have created an array of records from a list of dictionaries. We can see that the `x` field of `records` contains five [`numpy.int64`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int64) values:

```ipython3
records.x
```

If we wanted to look at each of the fields of `records`, we could pull them out individually from the array:

```ipython3
records.y
```

```ipython3
records.z
```

Clearly, for arrays with a large number of fields, retrieving each field in this manner would become tedious rather quickly. [`ak.unzip()`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip) can be used to directly build a tuple of the field arrays:

```ipython3
ak.unzip(records)
```

Records are not *required* to have field names. A record without field names is known as a “tuple”, e.g.

```ipython3
tuples = ak.Array(
    [
        (1, 1.1, "one"),
        (2, 2.2, "two"),
        (3, 3.3, "three"),
        (4, 4.4, "four"),
        (5, 5.5, "five"),
    ]
)
```

If we unzip an array of tuples, we obtain the same result as for records:

```ipython3
ak.unzip(tuples)
```

[`ak.unzip()`](sphinx-llm:a253bd50406941cf8e244c3a95347059#ak.unzip) can be combined with [`ak.fields()`](sphinx-llm:056b2c11871845eaa8962e15bfbffd9e#ak.fields) to build a mapping from field name to field array:

```ipython3
dict(zip(ak.fields(records), ak.unzip(records)))
```

For tuples, the field names will be strings corresponding to the field index:

```ipython3
dict(zip(ak.fields(tuples), ak.unzip(tuples)))
```

## Zipping together arrays

Because Awkward Arrays unzip into distinct arrays, it is reasonable to ask whether the reverse is possible, i.e. given the following arrays

```ipython3
age = ak.Array([18, 32, 87, 55])
name = ak.Array(["Dorit", "Caitlin", "Theodor", "Albano"]);
```

can we form an array of records? The [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) function provides a way to join compatible arrays into a single array of records:

```ipython3
people = ak.zip({"age": age, "name": name})
```

Similarly, we could also build an array of tuples by passing a sequence of arrays:

```ipython3
ak.zip([age, name])
```

Zipping and unzipping arrays is a lightweight operation, and so you should not hesitate to zip together arrays if it makes sense for the problem at hand. One of the benefits of combining arrays into an array of records is that slicing and masking operations are applied to all fields, e.g.

```ipython3
people[age > 35]
```

### Arrays with different dimensions

So far, we’ve looked at simple arrays with the same dimension in each field. It is actually possible to build arrays with fields of *different* dimensions, e.g.

```ipython3
x = ak.Array(
    [
        103,
        450,
        33,
        4,
    ]
)

digits_of_x = ak.Array(
    [
        [1, 0, 3],
        [4, 5, 0],
        [3, 3],
        [4],
    ]
)
x_and_digits = ak.zip({"x": x, "digits": digits_of_x})
```

The type of this array is

```ipython3
x_and_digits.type
```

Note that the `x` field has changed type:

```ipython3
x.type
```

```ipython3
x_and_digits.x.type
```

In zipping the two arrays together, the `x` has been broadcast against `digits_of_x`. Sometimes you might want to limit the broadcasting to a particular depth (dimension). This can be done by passing the `depth_limit` parameter:

```ipython3
x_and_digits = ak.zip({"x": x, "digits": digits_of_x}, depth_limit=1)
```

Now the `x` field has a single dimension

```ipython3
x_and_digits.x.type
```

### Arrays with different dimension lengths

What happens if we zip together arrays with the same dimensions, but different lengths in each dimensions?

```ipython3
x_and_y = ak.Array(
    [
        [103, 903],
        [450, 83],
        [33, 8],
        [4, 109],
    ]
)

digits_of_x_and_y = ak.Array(
    [
        [1, 0, 3, 9, 0, 3],
        [4, 5, 0, 8, 3],
        [3, 3, 8],
        [4, 1, 0, 9],
    ]
)

ak.zip({"x_and_y": x_and_y, "digits": digits_of_x_and_y})
```

```ipythontb
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[21], line 19
     15         [4, 1, 0, 9],
     16     ]
     17 )
     18 
---> 19 ak.zip({"x_and_y": x_and_y, "digits": digits_of_x_and_y})

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

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_zip.py:157, in zip(arrays, depth_limit, parameters, with_name, right_broadcast, optiontype_outside_record, highlevel, behavior, attrs)
    154     yield arrays
    156 # Implementation
--> 157 return _impl(
    158     arrays,
    159     depth_limit,
    160     parameters,
    161     with_name,
    162     right_broadcast,
    163     optiontype_outside_record,
    164     highlevel,
    165     behavior,
    166     attrs,
    167 )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/operations/ak_zip.py:251, in _impl(arrays, depth_limit, parameters, with_name, right_broadcast, optiontype_outside_record, highlevel, behavior, attrs)
    246         return None
    248 depth_context, lateral_context = NamedAxesWithDims.prepare_contexts(
    249     list(arrays.values()) if isinstance(arrays, Mapping) else list(arrays)
    250 )
--> 251 out = ak._broadcasting.broadcast_and_apply(
    252     layouts,
    253     action,
    254     depth_context=depth_context,
    255     lateral_context=lateral_context,
    256     right_broadcast=right_broadcast,
    257 )
    258 assert isinstance(out, tuple) and len(out) == 1
    259 out = out[0]

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

    ak.zip(
        {'x_and_y': <Array [[103, 903], [450, 83], [33, ...], [4, 109]] type=...
    )
```

Arrays which cannot be broadcast against each other will raise a `ValueError`. In this case, we want to stop broadcasting at the first dimension (`depth_limit=1`)

```ipython3
ak.zip({"x_and_y": x_and_y, "digits": digits_of_x_and_y}, depth_limit=1)
```

## Projecting arrays

Sometimes we are interested only in a subset of the fields of an array. For example, imagine that we have an array of coordinates on the $\hat{x}\hat{y}$ plane:

```ipython3
triangle = ak.Array(
    [
        {"x": 1, "y": 6, "z": 0},
        {"x": 2, "y": 7, "z": 0},
        {"x": 3, "y": 8, "z": 0},
    ]
)
```

If we know that these points should lie on a plane, then we might wish to discard the $\hat{z}$ coordinate. We can do this by slicing only the $\hat{x}$ and $\hat{y}$ fields:

```ipython3
triangle_2d = triangle[["x", "y"]]
```

Note that the key passed to the subscript operator is a [`list`](https://docs.python.org/3/library/stdtypes.html#list) `["x", "y"]`, not a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple). Awkward Array recognises the [`list`](https://docs.python.org/3/library/stdtypes.html#list) to mean “take both the `"x"` and `"y"` fields”.

Projections can be combined with array slicing and masking, e.g.

```ipython3
triangle_2d_first_2 = triangle[:2, ["x", "y"]]
```

Let’s now consider an array of triangles, i.e. a polygon:

```ipython3
triangles = ak.Array(
    [
        [
            {"x": 1, "y": 6, "z": 0},
            {"x": 2, "y": 7, "z": 0},
            {"x": 3, "y": 8, "z": 0},
        ],
        [
            {"x": 4, "y": 9, "z": 0},
            {"x": 5, "y": 10, "z": 0},
            {"x": 6, "y": 11, "z": 0},
        ],
    ]
)
```

We can combine an [`int`](https://docs.python.org/3/library/functions.html#int) index `0` with a [`str`](https://docs.python.org/3/library/stdtypes.html#str) projection to view the `"x"` coordinates of the first triangle vertices

```ipython3
triangles[0, "x"]
```

We could even ignore the first vertex of each triangle

```ipython3
triangles[0, 1:, "x"]
```

Projections *commute* (to the left) with other indices to produce the same result as their “natural” position. This means that the above projection could also be written as

```ipython3
triangles[0, "x", 1:]
```

or even

```ipython3
triangles["x", 0, 1:]
```

For columnar Awkward Arrays, there is no performance difference between any of these approaches; projecting the records of an array just changes its metadata, rather than invoking any loops over the data.

## Projecting records-of-records

The records of an array can themselves contain records

```ipython3
polygon = ak.Array(
    [
        {
            "vertex": [
                {"x": 1, "y": 6, "z": 0},
                {"x": 2, "y": 7, "z": 0},
                {"x": 3, "y": 8, "z": 0},
            ],
            "normal": [
                {"x": 0.164, "y": 0.986, "z": 0.0},
                {"x": 0.275, "y": 0.962, "z": 0.0},
                {"x": 0.351, "y": 0.936, "z": 0.0},
            ],
            "n_vertex": 3,
        },
        {
            "vertex": [
                {"x": 4, "y": 9, "z": 0},
                {"x": 5, "y": 10, "z": 0},
                {"x": 6, "y": 11, "z": 0},
                {"x": 7, "y": 12, "z": 0},
            ],
            "normal": [
                {"x": 0.406, "y": 0.914, "z": 0.0},
                {"x": 0.447, "y": 0.894, "z": 0.0},
                {"x": 0.470, "y": 0.878, "z": 0.0},
                {"x": 0.504, "y": 0.864, "z": 0.0},
            ],
            "n_vertex": 4,
        },
    ]
)
```

Naturally we can access the `"vertex"` field with the `.` operator:

```ipython3
polygon.vertex
```

We can view the `"x"` field of the vertex array with an additional lookup

```ipython3
polygon.vertex.x
```

The `.` operator represents the simplest slice of a single string, i.e.

```ipython3
polygon["vertex"]
```

The slice corresponding to the nested lookup `.vertex.x` is given by a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple) of [`str`](https://docs.python.org/3/library/stdtypes.html#str):

```ipython3
polygon[("vertex", "x")]
```

It is even possible to combine multiple and single projections. Let’s project the `"x"` field of the `"vertex"` and `"normal"` fields:

```ipython3
polygon[["vertex", "normal"], "x"]
```
