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

How to list an array's fields/columns/keys
==========================================

```{code-cell} ipython3
:tags: [hide-cell]

%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"
```

## Arrays of records

+++

As seen in {doc}`how-to-create-records`, one of Awkward Array's most useful features is the ability to compose separate arrays into a single record structure:

```{code-cell} ipython3
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

```{code-cell} ipython3
records.type
```

The {class}`ak.Array` object itself provides a convenient {attr}`ak.Array.fields` property that returns the list of field names

```{code-cell} ipython3
records.fields
```

In addition to this, Awkward Array also provides a high-level {func}`ak.fields` function that returns the same result

```{code-cell} ipython3
ak.fields(records)
```

## Arrays of tuples

+++

In addition to records, Awkward Array also has the concept of _tuples_.

```{code-cell} ipython3
tuples = ak.Array(
    [
        (1, 2, 3),
        (1, 2, 3),
    ]
)
```

These look very similar to records, but the fields are un-named:

```{code-cell} ipython3
tuples.type
```

Despite this, the {func}`ak.fields` function, and {attr}`ak.Array.fields` property both return non-empty lists of strings when used to query a tuple array:

```{code-cell} ipython3
ak.fields(tuples)
```

```{code-cell} ipython3
tuples.fields
```

The returned field names are string-quoted integers (`"0"`, `"1"`, ...) that refer to zero-indexed tuple _slots_, and can be used to project the array:

```{code-cell} ipython3
tuples["0"]
```

```{code-cell} ipython3
tuples["1"]
```

Whilst the fields of records can be accessed as attributes of the array:

```{code-cell} ipython3
records.x
```

The same is not true of tuples, because integers are not valid attribute names:

```{code-cell} ipython3
:tags: [raises-exception]

tuples.0
```

The close similarity between records and tuples naturally raises the question:
> How do I know whether an array contains records or tuples?

The {func}`ak.is_tuple` function can be used to differentiate between the two

```{code-cell} ipython3
ak.is_tuple(tuples)
```

```{code-cell} ipython3
ak.is_tuple(records)
```
