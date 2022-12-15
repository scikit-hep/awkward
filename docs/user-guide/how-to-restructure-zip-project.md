---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to restructure arrays with zip/unzip and project
====================================================

```{code-cell} ipython3
:tags: [hide-cell]

%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"
```

## Unzipping an array of records

As discussed in {doc}`how-to-create-records`, in addition to primitive types like {attr}`numpy.float64` and {class}`numpy.datetime64`, Awkward Arrays can also contain records. These records are formed from a fixed number of optionally named _fields_.

```{code-cell} ipython3
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

Although it is useful to be able to create arrays from a sequence of records (as [arrays of structures](https://en.wikipedia.org/wiki/AoS_and_SoA#Array_of_structures)), Awkward Array implements arrays as [_structures of arrays_](https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays). It is therefore more natural to think about arrays in terms of their fields. 
In the above example, we have created an array of records from a list of dictionaries. We can see that the `x` field of `records` contains five {attr}`numpy.int64` values:

```{code-cell} ipython3
records.x
```

If we wanted to look at each of the fields of `records`, we could pull them out individually from the array:

```{code-cell} ipython3
records.y
```

```{code-cell} ipython3
records.z
```

Clearly, for arrays with a large number of fields, retrieving each field in this manner would become tedious rather quickly. {func}`ak.unzip` can be used to directly build a tuple of the field arrays:

```{code-cell} ipython3
ak.unzip(records)
```

Records are not _required_ to have field names. A record without field names is known as a "tuple", e.g.

```{code-cell} ipython3
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

```{code-cell} ipython3
ak.unzip(tuples)
```

{func}`ak.unzip` can be combined with {func}`ak.fields` to build a mapping from field name to field array:

```{code-cell} ipython3
dict(zip(ak.fields(records), ak.unzip(records)))
```

For tuples, the field names will be strings corresponding to the field index:

```{code-cell} ipython3
dict(zip(ak.fields(tuples), ak.unzip(tuples)))
```

## Zipping together arrays
Because Awkward Arrays unzip into distinct arrays, it is reasonable to ask whether the reverse is possible, i.e. given the following arrays

```{code-cell} ipython3
age = ak.Array([18, 32, 87, 55])
name = ak.Array(["Dorit", "Caitlin", "Theodor", "Albano"]);
```

can we form an array of records? The {func}`ak.zip` function provides a way to join compatible arrays into a single array of records:

```{code-cell} ipython3
people = ak.zip({"age": age, "name": name})
```

Similarly, we could also build an array of tuples by passing a sequence of arrays:

```{code-cell} ipython3
ak.zip([age, name])
```

Zipping and unzipping arrays is a lightweight operation, and so you should not hesitate to zip together arrays if it makes sense for the problem at hand. One of the benefits of combining arrays into an array of records is that slicing and masking operations are applied to all fields, e.g.

```{code-cell} ipython3
people[age > 35]
```

### Arrays with different dimensions
So far, we've looked at simple arrays with the same dimension in each field. It is actually possible to build arrays with fields of _different_ dimensions, e.g.

```{code-cell} ipython3
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

```{code-cell} ipython3
x_and_digits.type
```

Note that the `x` field has changed type:

```{code-cell} ipython3
x.type
```

```{code-cell} ipython3
x_and_digits.x.type
```

In zipping the two arrays together, the `x` has been broadcast against `digits_of_x`. Sometimes you might want to limit the broadcasting to a particular depth (dimension). This can be done by passing the `depth_limit` parameter:

```{code-cell} ipython3
x_and_digits = ak.zip({"x": x, "digits": digits_of_x}, depth_limit=1)
```

Now the `x` field has a single dimension

```{code-cell} ipython3
x_and_digits.x.type
```

### Arrays with different dimension lengths
What happens if we zip together arrays with the same dimensions, but different lengths in each dimensions?

```{code-cell} ipython3
---
mystnb:
  execution_allow_errors: true
tags: [raises-exception]
---
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

Arrays which cannot be broadcast against each other will raise a `ValueError`. In this case, we want to stop broadcasting at the first dimension (`depth_limit=1`)

```{code-cell} ipython3
ak.zip({"x_and_y": x_and_y, "digits": digits_of_x_and_y}, depth_limit=1)
```

## Projecting arrays

Sometimes we are interested only in a subset of the fields of an array. For example, imagine that we have an array of coordinates on the {math}`\hat{x}\hat{y}` plane:

```{code-cell} ipython3
triangle = ak.Array(
    [
        {"x": 1, "y": 6, "z": 0},
        {"x": 2, "y": 7, "z": 0},
        {"x": 3, "y": 8, "z": 0},
    ]
)
```

If we know that these points should lie on a plane, then we might wish to discard the {math}`\hat{z}` coordinate. We can do this by slicing only the {math}`\hat{x}` and {math}`\hat{y}` fields:

```{code-cell} ipython3
triangle_2d = triangle[["x", "y"]]
```

Note that the key passed to the subscript operator is a {class}`list` `["x", "y"]`, not a {class}`tuple`. Awkward Array recognises the {class}`list` to mean "take both the `"x"` and `"y"` fields".

+++

Projections can be combined with array slicing and masking, e.g.

```{code-cell} ipython3
triangle_2d_first_2 = triangle[:2, ["x", "y"]]
```

Let's now consider an array of triangles, i.e. a polygon:

```{code-cell} ipython3
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

We can combine an {class}`int` index `0` with a {class}`str` projection to view the `"x"` coordinates of the first triangle vertices

```{code-cell} ipython3
triangles[0, "x"]
```

We could even ignore the first vertex of each triangle

```{code-cell} ipython3
triangles[0, 1:, "x"]
```

Projections _commute_ (to the left) with other indices to produce the same result as their "natural" position. This means that the above projection could also be written as

```{code-cell} ipython3
triangles[0, "x", 1:]
```

or even

```{code-cell} ipython3
triangles["x", 0, 1:]
```

For columnar Awkward Arrays, there is no performance difference between any of these approaches; projecting the records of an array just changes its metadata, rather than invoking any loops over the data.

+++

## Projecting records-of-records

+++

The records of an array can themselves contain records

```{code-cell} ipython3
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

```{code-cell} ipython3
polygon.vertex
```

We can view the `"x"` field of the vertex array with an additional lookup

```{code-cell} ipython3
polygon.vertex.x
```

The `.` operator represents the simplest slice of a single string, i.e.

```{code-cell} ipython3
polygon["vertex"]
```

The slice corresponding to the nested lookup `.vertex.x` is given by a {class}`tuple` of {class}`str`:

```{code-cell} ipython3
polygon[("vertex", "x")]
```

It is even possible to combine multiple and single projections. Let's project the `"x"` field of the `"vertex"` and `"normal"` fields:

```{code-cell} ipython3
polygon[["vertex", "normal"], "x"]
```
