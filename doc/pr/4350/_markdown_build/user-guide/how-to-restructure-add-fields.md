# How to restructure arrays by adding fields

```ipython3
import awkward as ak
import numpy as np
```

## Adding fields to existing arrays

### Using `array['x']`

[How to examine an array with simple slicing](sphinx-llm:0fac06d97a01408abff982e8bfe38f9c) describes the wide variety of [`slice`](https://docs.python.org/3/library/functions.html#slice) types that can be used to pull values out of an Awkward Array. However, only single field-slicing is supported for *assignment* of new values.

```ipython3
array = ak.Array({"x": [1, 2, 3]})
array.show()
```

```myst-ansi
[{x: 1},
 {x: 2},
 {x: 3}]
```

To assign a new value to an existing array, we can simply use the subscript operator with the string name of the field. For example, to set the `x` field, we can write

```ipython3
array["x"] = [-1, -2, 3]
array.show()
```

```myst-ansi
[{x: -1},
 {x: -2},
 {x: 3}]
```

This might seem strange, given that we describe Awkward Arrays as *immutable*. A more detailed explaination is given in the [Advanced Users](sphinx-llm:22d692f066064fbda62a48fa56c37432) call-out, but it suffices to say that the *fields* of an array can be replaced, but individual values within an array cannot.

<a id="admonition-immutable-arrays"></a>

Using this syntax, we can assign to a *new* field of an array:

```ipython3
array["y"] = [9, 8, 7]
array.show()
```

```myst-ansi
[{x: -1, y: 9},
 {x: -2, y: 8},
 {x: 3, y: 7}]
```

If necessary, the new field will be broadcasted to fit the array. For example, we can introduce a third field `z` that is set to the constant `0`:

```ipython3
array["z"] = 0
array.show()
```

```myst-ansi
[{x: -1, y: 9, z: 0},
 {x: -2, y: 8, z: 0},
 {x: 3, y: 7, z: 0}]
```

A field can also be assigned deeply into a nested record e.g.

```ipython3
nested = ak.zip({"a": ak.zip({"x": [1, 2, 3]})})
nested["a", "y"] = 2 * nested.a.x

nested.show()
```

```myst-ansi
[{a: {x: 1, y: 2}},
 {a: {x: 2, y: 4}},
 {a: {x: 3, y: 6}}]
```

Note that the following does **not** work:

```ipython3
nested["a"]["y"] = 2 * nested.a.x  # does not work, nested["a"] is a copy!
nested.show()
```

```myst-ansi
[{a: {x: 1, y: 2}},
 {a: {x: 2, y: 4}},
 {a: {x: 3, y: 6}}]
```

Why does this happen? Well, Python first evaluates `nested["a"]`, which returns a *new* [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array) that is a (shallow) copy of the data in `nested.a`. Hence, the next step — to set `y` — operates on a *different*  [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array), and `nested.a` remains unchanged. The [Advanced Users](sphinx-llm:22d692f066064fbda62a48fa56c37432) call-out provides a more detailed explanation for *why* this does not work.

### Using `ak.with_field`

Sometimes you might not want to modify an existing array, but rather produce a new array with the new field. Whilst this can be done using a shallow copy, e.g.

```ipython3
import copy

copied = copy.copy(nested)
copied["z"] = [10, 20, 30]

copied.show()
```

```myst-ansi
[{a: {x: 1, y: 2}, z: 10},
 {a: {x: 2, y: 4}, z: 20},
 {a: {x: 3, y: 6}, z: 30}]
```

```ipython3
nested.show()
```

```myst-ansi
[{a: {x: 1, y: 2}},
 {a: {x: 2, y: 4}},
 {a: {x: 3, y: 6}}]
```

Awkward provides a dedicated function [`ak.with_field()`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field) that does this.

#### NOTE
Setting a field with `array['x']` uses [`ak.with_field()`](sphinx-llm:3c2fe1ca08b646e494d171d5ea7f2e33#ak.with_field) under the hood, so performance is not a factor in choosing one over the other.
