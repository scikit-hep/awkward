# How to create arrays of records

In Awkward Array, a “record” is a structure containing a fixed-length set of typed, possibly named fields. This is a “struct” in C or an “object” in Python (though the association of executable methods to record types is looser than the binding of methods to classes in object oriented languages).

All methods in Awkward Array are implemented as “[structs of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA),” rather than arrays of structs, so making and breaking records are inexpensive operations that you can perform frequently in data analysis.

```ipython3
import awkward as ak
import numpy as np
```

## From a list of Python dicts

Records have a natural representation in JSON and Python as dicts, but only if all dicts in a series have the same set of field names. The [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) invokes [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) whenever presented with a list (or other non-string, non-dict iterable).

```ipython3
python_dicts = [
    {"x": 1, "y": 1.1, "z": "one"},
    {"x": 2, "y": 2.2, "z": "two"},
    {"x": 3, "y": 3.3, "z": "three"},
    {"x": 4, "y": 4.4, "z": "four"},
    {"x": 5, "y": 5.5, "z": "five"},
]
python_dicts
```

It is important that all of the dicts in the series have the same set of field names, since Awkward Array has to identify all of the records as having a single type:

```ipython3
awkward_array = ak.Array(python_dicts)
awkward_array
```

That is to say, an array of records is not a mapping from one type to another, such as from strings to numbers. The above record has exactly three fields: *x*, *y*, and *z*, and they have fixed types: `int64`, `float64`, and `string`. A mapping could have a variable set of keys, but the value type would have to be uniform.

If you *try* to mix field types in [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) (ultimately from [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder)), the union of all sets of fields will be assumed, and any that aren’t filled in every item will be presumed “missing.”

```ipython3
array = ak.Array(
    [
        {"a": 1, "b": 1, "c": 1},
        {"b": 2, "c": 2},
        {"c": 3, "d": 3, "e": 3, "f": 3},
        {"c": 4},
    ]
)
array
```

## From a single dict of columns

If a *single dict* is passed to [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array), it will be interpreted as a set of columns, like [Pandas’s DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) constructor. This is to provide a familiar interface to Pandas users.

```ipython3
from_columns = ak.Array(
    {
        "x": [1, 2, 3, 4, 5],
        "y": [1.1, 2.2, 3.3, 4.4, 5.5],
        "z": ["one", "two", "three", "four", "five"],
    }
)
from_columns
```

This is *not* the same as calling [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) on the same input, which could not be a valid [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) because a single dict would be an [`ak.Record`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record).

```ipython3
from_rows = ak.from_iter(
    {
        "x": [1, 2, 3, 4, 5],
        "y": [1.1, 2.2, 3.3, 4.4, 5.5],
        "z": ["one", "two", "three", "four", "five"],
    }
)
from_rows
```

## Using ak.zip

The [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) function combines columns into an array of records, similar to the Pandas-style constructor described above.

```ipython3
from_columns = ak.zip(
    {
        "x": [1, 2, 3, 4, 5],
        "y": [1.1, 2.2, 3.3, 4.4, 5.5],
        "z": ["one", "two", "three", "four", "five"],
    }
)
from_columns
```

The difference is that [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) attempts to nested lists deeply, up to a `depth_limit`.

Given columns with nested lists:

```ipython3
zipped = ak.zip(
    {
        "x": ak.Array([[1, 2, 3], [], [4, 5]]),
        "y": ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
    }
)
zipped
```

By contrast, the same input to [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array)’s Pandas-style constructor keeps nested lists separate.

```ipython3
not_zipped = ak.Array(
    {
        "x": ak.Array([[1, 2, 3], [], [4, 5]]),
        "y": ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
    }
)
not_zipped
```

The difference can be seen in a comparison of their types:

```ipython3
zipped.type.show()
not_zipped.type.show()
```

```myst-ansi
3 * var * {
    x: int64,
    y: float64
}
3 * {
    x: var * int64,
    y: var * float64
}
```

Also, [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) can build records without field names, also known as “tuples.”

```ipython3
tuples = ak.zip(
    (ak.Array([[1, 2, 3], [], [4, 5]]), ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
)
tuples
```

Functions that return lists of pairs, such as [`ak.cartesian()`](sphinx-llm:e515482efbae4eb28bb9f06e625d21c4#ak.cartesian) and [`ak.combinations()`](sphinx-llm:e2edae471dd3475f90dd9ae86f4e03a3#ak.combinations), also use the tuple type.

```ipython3
one = ak.Array([[1, 2, 3], [], [4, 5], [6]])
two = ak.Array([["a", "b"], ["c"], ["d"], ["e", "f"]])

ak.cartesian((one, two))
```

```ipython3
ak.combinations(one, 2)
```

## Record names

In addition to optional field names, record types can also have names.

```ipython3
ak.Array(
    [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
        {"x": 4, "y": 4.4},
        {"x": 5, "y": 5.5},
    ],
    with_name="XYZ",
)
```

```ipython3
ak.zip(
    (ak.Array([[1, 2, 3], [], [4, 5]]), ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])),
    with_name="AB",
)
```

```ipython3
ak.cartesian(
    {"L": ak.Array([1, 2, 3]), "R": ak.Array(["a", "b"])}, with_name="LeftRight", axis=0
)
```

Names are for giving records specialized behavior through the [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) registry (see the [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) reference documentation for details). These are like attaching methods to a class in the sense that all records with a particular name can be given Python properties and methods.

```ipython3
class XYZRecord(ak.Record):
    def __repr__(self):
        return f"(X={self.x}:Y={self.y})"


class XYZArray(ak.Array):
    def diff(self):
        return abs(self.x - self.y)


ak.behavior["XYZ"] = XYZRecord
ak.behavior["*", "XYZ"] = XYZArray
ak.behavior["__typestr__", "XYZ"] = "XYZ"
ak.behavior[np.sqrt, "XYZ"] = lambda xyz: np.sqrt(xyz.x)

array = ak.Array(
    [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
        {"x": 4, "y": 4.4},
        {"x": 5, "y": 5.5},
    ],
    with_name="XYZ",
)
array
```

```ipython3
array.diff()
```

```ipython3
array[0]
```

```ipython3
np.sqrt(array)
```

## With ArrayBuilder

[`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) is described in more detail [in this tutorial](sphinx-llm:26cb6d67e3ae4372aacb2b87a0852a58), but you can also construct arrays of records using the `begin_record`/`end_record` methods or the `record` context manager.

(This is what [`ak.from_iter()`](sphinx-llm:40c924e275044a0993aef7f20508d933#ak.from_iter) uses internally to accumulate records.)

```ipython3
builder = ak.ArrayBuilder()

builder.begin_record()
builder.field("x").append(1)
builder.field("y").append(1.1)
builder.end_record()

builder.begin_record()
builder.field("x").append(2)
builder.field("y").append(2.2)
builder.end_record()

builder.begin_record()
builder.field("x").append(3)
builder.field("y").append(3.3)
builder.end_record()

array = builder.snapshot()
array
```

```ipython3
builder = ak.ArrayBuilder()

with builder.record():
    builder.field("x").append(1)
    builder.field("y").append(1.1)

with builder.record():
    builder.field("x").append(2)
    builder.field("y").append(2.2)

with builder.record():
    builder.field("x").append(3)
    builder.field("y").append(3.3)

array = builder.snapshot()
array
```

The `begin_record` method and `record` context manager can take a name, which allows you to name your records on the spot.

```ipython3
builder = ak.ArrayBuilder()

for i in range(3):
    with builder.record("XY"):
        builder.field("x").append(i)
        builder.field("y").append(i * 1.1)

array = builder.snapshot()
array
```

Record names also let you decide between two ways of dealing with changing sets of fields among the input records.

* The default/generic way: all records at a given level are taken to belong to the same type and some instances of that type are missing fields.
* Named records: different names mean different types, whether they have the same fields or not. Different record names at the same level result in a union type.

Here is an illustration of that difference.

```ipython3
def fill_A(builder, use_name):
    with builder.record("A" if use_name else None):
        builder.field("x").append(len(builder))
        builder.field("y").append(len(builder) * 1.1)


def fill_B(builder, use_name):
    with builder.record("B" if use_name else None):
        builder.field("y").append(len(builder) * 1.1)
        builder.field("z").append(str(len(builder)))
```

Without names:

```ipython3
builder = ak.ArrayBuilder()

fill_A(builder, use_name=False)
fill_A(builder, use_name=False)

fill_B(builder, use_name=False)
fill_B(builder, use_name=False)

fill_A(builder, use_name=False)

without_names = builder.snapshot()
without_names
```

With names:

```ipython3
builder = ak.ArrayBuilder()

fill_A(builder, use_name=True)
fill_A(builder, use_name=True)

fill_B(builder, use_name=True)
fill_B(builder, use_name=True)

fill_A(builder, use_name=True)

with_names = builder.snapshot()
with_names
```

The difference can be seen in the type: `without_names` has only one record type, but the *x* and *z* fields are optional, and `with_names` has a union of two record types, neither of which have optional fields.

```ipython3
without_names.type.show()
```

```myst-ansi
5 * {
    x: ?int64,
    y: float64,
    z: ?string
}
```

```ipython3
with_names.type.show()
```

```myst-ansi
5 * union[
    A[
        x: int64,
        y: float64
    ],
    B[
        y: float64,
        z: string
    ]
]
```

## In Numba

Functions that Numba Just-In-Time (JIT) compiles can use [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) to accumulate records or columns can be accumulated in the compiled part and later combined with [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip). Combining columns outside of a JIT-compiled function is generally faster.

([At this time](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#language), Numba can’t use context managers, the `with` statement, in fully compiled code. [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) can’t be constructed or converted to an array using `snapshot` inside a JIT-compiled function, but can be outside the compiled context. Similarly, `ak.*` functions like [`ak.zip()`](sphinx-llm:8d4df9d3d238429d957dd1efd0afa844#ak.zip) can’t be called inside a JIT-compiled function, but can be outside.)

```ipython3
import numba as nb
```

```ipython3
@nb.jit
def append_record(builder, i):
    builder.begin_record()
    builder.field("x").append(i)
    builder.field("y").append(i * 1.1)
    builder.end_record()


@nb.jit
def example(builder):
    append_record(builder, 1)
    append_record(builder, 2)
    append_record(builder, 3)
    return builder


builder = example(ak.ArrayBuilder())

array = builder.snapshot()
array
```

```ipython3
@nb.jit
def faster_example():
    x = np.empty(3, np.int64)
    y = np.empty(3, np.float64)
    x[0] = 1
    y[0] = 1.1
    x[1] = 2
    y[1] = 2.2
    x[2] = 3
    y[2] = 3.3
    return x, y


array = ak.zip(dict(zip(["x", "y"], faster_example())))
array
```

Combining columns into arrays is a metadata-only operation, which does not scale with the size of the dataset. The second example is faster because it only requires Numba to fill NumPy arrays, which it’s designed for, without involving the machinery of [`ak.ArrayBuilder`](sphinx-llm:2fb9ffd5c3a441658693b9cc500d5518#ak.ArrayBuilder) to identify types at runtime.
