---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to examine an array's type
==============================

The _type_ of an Awkward Array can be determined using the {func}`ak.type` function, or {attr}`ak.Array.type` attribute of an array. It describes both the data-types of an array, e.g. `float64`, and the structure of the array (how many dimensions, which dimensions are ragged, which dimensions contain missing values, etc.).

+++

(how-to-examine-type:array-types)=
## Array types

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

{attr}`ak.Array.type` always returns an {class}`ak.types.ArrayType` object describing the outermost length of the array, which is always known.[^tt] The {class}`ak.types.ArrayType` wraps a {class}`ak.types.Type` object, which represents an array of "something". For example, an array of integers:
[^tt]: Except for typetracer arrays, which are used in the [dask-awkward](https://github.com/dask-contrib/dask-awkward) integration.

```{code-cell} ipython3
ak.Array([1, 2, 3]).type
```

The outermost {class}`ak.types.ArrayType` object indicates that this array has a known length of 3. Its content

```{code-cell} ipython3
ak.Array([1, 2, 3]).type.content
```

describes the array itself, which is an array of {data}`np.int64`.

+++

### Regular vs ragged dimensions

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

### Records and tuples

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

### Missing items

Missing items are represented by both the `option[...]` and `?` tokens, according to readability:

```{code-cell} ipython3
missing = ak.Array([33.0, None, 15.5, 99.1])
missing.type.show()
```

Awkward's {class}`ak.types.OptionType` object is used to represent this datashape type:

```{code-cell} ipython3
missing.type
```

### Unions

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

### Strings

+++

Awkward Array implements strings as views over a 1D array of `uint8` characters (`char`):

```{code-cell} ipython3
ak.type("hello world")
```

This concept extends to an array of strings:

```{code-cell} ipython3
array = ak.Array(
    ["Mr.", "Blue,", "you", "did", "it", "right"]
)
array.type
```

`array` is a list of strings, which is represented as a list-of-list-of-char. When we evaluate `str(array.type)` (or directly print this value with `array.type.show()`), Awkward returns a readable type-string:

```{code-cell} ipython3
array.type.show()
```

## Scalar types

In {ref}`how-to-examine-type:array-types` it was discussed that all {class}`ak.type.Type` objects are array-types, e.g. {class}`ak.types.NumpyType` is the type of a NumPy (or CuPy, etc.) array of a fixed dtype:

```{code-cell} ipython3
import numpy as np

ak.type(np.arange(3))
```

Let's now consider the following array of records:

```{code-cell} ipython3
record_array = ak.Array([
    {'x': 10, 'y': 11}
])
record_array.type
```

The resulting type object is an {class}`ak.types.ArrayType` of {class}`ak.types.RecordType`. This record-type represents an array of records, built from two NumPy arrays. From outside-to-inside, we can read the type object as:
- An array of length 1
- that is an array of records with two fields 'x' and 'y'
- which are both NumPy arrays of {data}`np.int64` type.

Now, what happens if we pull out a single record and inspect its type?

```{code-cell} ipython3
record = record_array[0]
record.type
```

Unlike the {class}`ak.types.ArrayType` objects returned by {func}`ak.type` for arrays, {attr}`ak.Record.type` always returns a {class}`ak.types.ScalarType` object. Reading the returned type again from outside-to-inside, we have
- A scalar taken from an array
- that is an array of records with two fields 'x' and 'y'
- which are both NumPy arrays of {data}`np.int64` type.

Like {class}`ak.types.ArrayType`, {class}`ak.types.ScalarType` is an _outermost_ type, but unlike {class}`ak.types.ArrayType` it does more than add length information; it also _removes a dimension_ from the final type!
