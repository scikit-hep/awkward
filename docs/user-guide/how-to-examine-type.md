---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to examine an array's type
==============================

+++

The type of an Awkward Array can be found with the {func}`ak.type` function, or {attr}`ak.Array.type` attribute of an array. It describes both the _data_ types of an array, e.g. `float64`, and the structure of the array (how many dimensions, which dimensions are ragged, which dimensions contain missing values, etc.).

```{code-cell} ipython3
import awkward as ak

array = ak.Array(
    [
        ["Mr.", "Blue,", "you", "did", "it", "right"],
        ["But", "soon", "comes", "Mr.", "Night"],
        ["creepin'", "over"],
    ]
)
array.type.show()
```

`array.type.show()` displays an extended subset of the [Datashape](https://datashape.readthedocs.io/en/latest/overview.html) language, which describes both shape and layout of an array in the form of _units_ and _dimensions_. `array.type` actually returns an {class}`ak.types.Type` object, which can be inspected

```{code-cell} ipython3
array.type
```

```{code-cell} ipython3
array.type.length
```

From inspecting `array.type`, we can see that Awkward Array implements strings as views over a 1D array of `uint8` characters:

```{code-cell} ipython3
array.type.content.content.content
```

{attr}`ak.Array.type` always returns an {class}`ak.types.ArrayType` object describing the outermost length of the array, which is always known.[^tt]

[^tt]: Except for typetracer arrays, which are used in the [dask-awkward](https://github.com/dask-contrib/dask-awkward) integration.

+++

## Regular vs ragged dimensions

+++

Regular arrays and ragged arrays have different types

```{code-cell} ipython3
import numpy as np

regular = ak.from_numpy(np.arange(8).reshape(2, 4))
ragged = ak.from_regular(regular)

regular.type.show()
ragged.type.show()
```

In the Datashape language, ragged dimensions are described as `var`, whilst regular (`fixed`) dimensions are expressed by an integer representing their size. At the type level, the `ragged` type object does not contain any size information, as it is no longer a constant part of the type:

```{code-cell} ipython3
regular.type.content.size
```

```{code-cell} ipython3
:tags: [raises-exception]

ragged.type.content.size
```

## Records and tuples

+++

An Awkward Array with records is expressed using curly braces, resembling a JSON object or Python dictionary:

```{code-cell} ipython3
poet_records = ak.Array(
    [
        {"first": "William", "last": "Shakespeare"},
        {"first": "Sylvia", "last": "Plath"},
        {"first": "Homer", "last": "Simpson"},
    ]
)

poet_records.type.show()
```

whereas an array with tuples is expressed using parentheses, resembling a Python tuple:

```{code-cell} ipython3
poet_tuples = ak.Array(
    [
        ("William", "Shakespeare"),
        ("Sylvia", "Plath"),
        ("Homer", "Simpson"),
    ]
)

poet_tuples.type.show()
```

The {class}`ak.types.RecordType` object contains information such as whether the record is a tuple, e.g.

```{code-cell} ipython3
poet_records.type.content.is_tuple
```

```{code-cell} ipython3
poet_tuples.type.content.is_tuple
```

Let's look at the type of a simpler array:

```{code-cell} ipython3
ak.type([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
```

## Missing items

+++

Missing items are represented by both the `option[...]` and `?` tokens, according to readability:

```{code-cell} ipython3
missing = ak.Array([33.0, None, 15.5, 99.1])
missing.type.show()
```

Awkward's {class}`ak.types.OptionType` object is used to represent this datashape type:

```{code-cell} ipython3
missing.type.content
```

## Unions

+++

A union is formed whenever multiple types are required for a particular dimension, e.g. if we concatenate two arrays with different records:

```{code-cell} ipython3
mixed = ak.concatenate(
    (
        [{"x": 1}],
        [{"y": 2}],
    )
)
mixed.type.show()
```

From the printed type, we can see that the formed union has two possible types. We can inspect these from the {class}`ak.types.UnionType` object in `mixed.type.content`

```{code-cell} ipython3
mixed.type.content
```

```{code-cell} ipython3
mixed.type.content.contents[0].show()
```

```{code-cell} ipython3
mixed.type.content.contents[1].show()
```
