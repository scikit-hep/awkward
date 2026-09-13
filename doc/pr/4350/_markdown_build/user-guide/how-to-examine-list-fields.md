# How to list an array’s fields/columns/keys

```ipython3
%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"
```

## Arrays of records

As seen in [How to create arrays of records](sphinx-llm:e2a08ad149184f49b0882e363831b487), one of Awkward Array’s most useful features is the ability to compose separate arrays into a single record structure:

```ipython3
import awkward as ak
import numpy as np

records = ak.Array(
    [
        {"x": 0.014309631995020777, "y": 0.7077380205549498},
        {"x": 0.44925764718311145, "y": 0.11927022136408238},
        {"x": 0.9870653236436898, "y": 0.1543661194285082},
        {"x": 0.7071893130949595, "y": 0.3966721033002645},
        {"x": 0.3059032831996634, "y": 0.5094743992919755},
    ]
)
```

The type of an array gives an indication of the fields that it contains. We can see that the `records` array contains two fields `"x"` and `"y"`:

```ipython3
print(records.type)
```

```myst-ansi
5 * {x: float64, y: float64}
```

```ipython3
records.type.show()
```

```myst-ansi
5 * {
    x: float64,
    y: float64
}
```

The [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) object itself provides a convenient [`ak.Array.fields`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.fields) property that returns the list of field names

```ipython3
records.fields
```

In addition to this, Awkward Array also provides a high-level [`ak.fields()`](sphinx-llm:056b2c11871845eaa8962e15bfbffd9e#ak.fields) function that returns the same result

```ipython3
ak.fields(records)
```

## Arrays of tuples

In addition to records, Awkward Array also has the concept of *tuples*.

```ipython3
tuples = ak.Array(
    [
        (1, 2, 3),
        (1, 2, 3),
    ]
)
```

These look very similar to records, but the fields are un-named:

```ipython3
print(tuples.type)
```

```myst-ansi
2 * (int64, int64, int64)
```

Despite this, the [`ak.fields()`](sphinx-llm:056b2c11871845eaa8962e15bfbffd9e#ak.fields) function, and [`ak.Array.fields`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.fields) property both return non-empty lists of strings when used to query a tuple array:

```ipython3
ak.fields(tuples)
```

```ipython3
tuples.fields
```

The returned field names are string-quoted integers (`"0"`, `"1"`, …) that refer to zero-indexed tuple *slots*, and can be used to project the array:

```ipython3
tuples["0"]
```

```ipython3
tuples["1"]
```

Whilst the fields of records can be accessed as attributes of the array:

```ipython3
records.x
```

The same is not true of tuples, because integers are not valid attribute names:

```ipython3
tuples.0
```

```ipythontb
  Cell In[14], line 1
    tuples.0
          ^
SyntaxError: invalid syntax
```

The close similarity between records and tuples naturally raises the question:

> How do I know whether an array contains records or tuples?

The [`ak.is_tuple()`](sphinx-llm:8bb0c04aeecd42e59006ba23c707a6f4#ak.is_tuple) function can be used to differentiate between the two

```ipython3
ak.is_tuple(tuples)
```

```ipython3
ak.is_tuple(records)
```
