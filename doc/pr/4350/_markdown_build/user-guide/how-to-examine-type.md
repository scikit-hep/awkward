# How to examine an array’s type

The *type* of an Awkward Array can be determined using the [`ak.type()`](sphinx-llm:65d3dedfc8ad48c98f3bacc6472ae8c8#ak.type) function, or [`ak.Array.type`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.type) attribute of an array. It describes both the data-types of an array, e.g. `float64`, and the structure of the array (how many dimensions, which dimensions are ragged, which dimensions contain missing values, etc.).

<a id="how-to-examine-type-array-types"></a>

## Array types

```ipython3
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

```myst-ansi
3 * var * string
```

`array.type.show()` displays an extended subset of the [Datashape](https://datashape.readthedocs.io/en/latest/overview.html) language, which describes both shape and layout of an array in the form of *units* and *dimensions*. `array.type` actually returns an [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type) object, which can be inspected

```ipython3
array.type
```

[`ak.Array.type`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.type) always returns an [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType) object describing the outermost length of the array, which is always known.<sup>[1](#tt)</sup> The [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType) wraps a [`ak.types.Type`](sphinx-llm:fc995ece0ffa4f68adbe74567c7b2ea1#ak.types.Type) object, which represents an array of “something”. For example, an array of integers:

```ipython3
ak.Array([1, 2, 3]).type
```

The outermost [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType) object indicates that this array has a known length of 3. Its content

```ipython3
ak.Array([1, 2, 3]).type.content
```

describes the array itself, which is an array of `np.int64`.

### Regular vs ragged dimensions

Regular arrays and ragged arrays have different types

```ipython3
import numpy as np

regular = ak.from_numpy(np.arange(8).reshape(2, 4))
ragged = ak.from_regular(regular)

regular.type.show()
ragged.type.show()
```

```myst-ansi
2 * 4 * int64
2 * var * int64
```

In the Datashape language, ragged dimensions are described as `var`, whilst regular (`fixed`) dimensions are expressed by an integer representing their size. At the type level, the `ragged` type object does not contain any size information, as it is no longer a constant part of the type:

```ipython3
regular.type.content.size
```

```ipython3
ragged.type.content.size
```

```ipythontb
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[7], line 1
----> 1 ragged.type.content.size

AttributeError: 'ListType' object has no attribute 'size'
```

### Records and tuples

An Awkward Array with records is expressed using curly braces, resembling a JSON object or Python dictionary:

```ipython3
poet_records = ak.Array(
    [
        {"first": "William", "last": "Shakespeare"},
        {"first": "Sylvia", "last": "Plath"},
        {"first": "Homer", "last": "Simpson"},
    ]
)

poet_records.type.show()
```

```myst-ansi
3 * {
    first: string,
    last: string
}
```

whereas an array with tuples is expressed using parentheses, resembling a Python tuple:

```ipython3
poet_tuples = ak.Array(
    [
        ("William", "Shakespeare"),
        ("Sylvia", "Plath"),
        ("Homer", "Simpson"),
    ]
)

poet_tuples.type.show()
```

```myst-ansi
3 * (
    string,
    string
)
```

The [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType) object contains information such as whether the record is a tuple, e.g.

```ipython3
poet_records.type.content.is_tuple
```

```ipython3
poet_tuples.type.content.is_tuple
```

Let’s look at the type of a simpler array:

```ipython3
ak.type([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
```

### Missing items

Missing items are represented by both the `option[...]` and `?` tokens, according to readability:

```ipython3
missing = ak.Array([33.0, None, 15.5, 99.1])
missing.type.show()
```

```myst-ansi
4 * ?float64
```

Awkward’s [`ak.types.OptionType`](sphinx-llm:f8e0eb36e1334eef887fb5ab81458600#ak.types.OptionType) object is used to represent this datashape type:

```ipython3
missing.type
```

### Unions

A union is formed whenever multiple types are required for a particular dimension, e.g. if we concatenate two arrays with different records:

```ipython3
mixed = ak.concatenate(
    (
        [{"x": 1}],
        [{"y": 2}],
    )
)
mixed.type.show()
```

```myst-ansi
2 * union[
    {
        x: int64
    },
    {
        y: int64
    }
]
```

From the printed type, we can see that the formed union has two possible types. We can inspect these from the [`ak.types.UnionType`](sphinx-llm:5e1485b00d7b47f39f0082bb755fa92f#ak.types.UnionType) object in `mixed.type.content`

```ipython3
mixed.type.content
```

```ipython3
mixed.type.content.contents[0].show()
```

```myst-ansi
{
    x: int64
}
```

```ipython3
mixed.type.content.contents[1].show()
```

```myst-ansi
{
    y: int64
}
```

### Strings

Awkward Array implements strings as views over a 1D array of `uint8` characters (`char`):

```ipython3
ak.type("hello world")
```

This concept extends to an array of strings:

```ipython3
array = ak.Array(
    ["Mr.", "Blue,", "you", "did", "it", "right"]
)
array.type
```

`array` is a list of strings, which is represented as a list-of-list-of-char. When we evaluate `str(array.type)` (or directly print this value with `array.type.show()`), Awkward returns a readable type-string:

```ipython3
array.type.show()
```

```myst-ansi
6 * string
```

## Scalar types

In [Array types](sphinx-llm:e8d7535ce8844eb7a03b602160f4ffed) it was discussed that all `ak.type.Type` objects are array-types, e.g. [`ak.types.NumpyType`](sphinx-llm:7284cac2045946428822c632c80c9067#ak.types.NumpyType) is the type of a NumPy (or CuPy, etc.) array of a fixed dtype:

```ipython3
import numpy as np

ak.type(np.arange(3))
```

Let’s now consider the following array of records:

```ipython3
record_array = ak.Array([
    {'x': 10, 'y': 11}
])
record_array.type
```

The resulting type object is an [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType) of [`ak.types.RecordType`](sphinx-llm:516b58d6a2f7412aadbaf0859053019a#ak.types.RecordType). This record-type represents an array of records, built from two NumPy arrays. From outside-to-inside, we can read the type object as:

- An array of length 1
- that is an array of records with two fields ‘x’ and ‘y’
- which are both NumPy arrays of `np.int64` type.

Now, what happens if we pull out a single record and inspect its type?

```ipython3
record = record_array[0]
record.type
```

Unlike the [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType) objects returned by [`ak.type()`](sphinx-llm:65d3dedfc8ad48c98f3bacc6472ae8c8#ak.type) for arrays, [`ak.Record.type`](sphinx-llm:e158df99f80043309c758f6e933352b2#ak.Record.type) always returns a [`ak.types.ScalarType`](sphinx-llm:ad298acb59fd4c1bb1bf103171505226#ak.types.ScalarType) object. Reading the returned type again from outside-to-inside, we have

- A scalar taken from an array
- that is an array of records with two fields ‘x’ and ‘y’
- which are both NumPy arrays of `np.int64` type.

Like [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType), [`ak.types.ScalarType`](sphinx-llm:ad298acb59fd4c1bb1bf103171505226#ak.types.ScalarType) is an *outermost* type, but unlike [`ak.types.ArrayType`](sphinx-llm:53688a8c7ce14810a6b2ed10d56f008f#ak.types.ArrayType) it does more than add length information; it also *removes a dimension* from the final type!

---
* <a id='tt'>**[1]**</a> Except for typetracer arrays, which are used in the [dask-awkward](https://github.com/dask-contrib/dask-awkward) integration.
