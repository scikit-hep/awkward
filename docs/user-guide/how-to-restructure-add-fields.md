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

How to restructure arrays by adding fields
==========================================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## Adding fields to existing arrays

+++

### Using `array['x']`

+++

{doc}`how-to-examine-simple-slicing` describes the wide variety of {class}`slice` types that can be used to pull values out of an Awkward Array. However, only single field-slicing is supported for _assignment_ of new values.

```{code-cell} ipython3
array = ak.Array({"x": [1, 2, 3]})
array.show()
```

To assign a new value to an existing array, we can simply use the subscript operator with the string name of the field. For example, to set the `x` field, we can write

```{code-cell} ipython3
array["x"] = [-1, -2, 3]
array.show()
```

This might seem strange, given that we describe Awkward Arrays as _immutable_. A more detailed explaination is given in the {ref}`Advanced Users<admonition:immutable-arrays>` call-out, but it suffices to say that the _fields_ of an array can be replaced, but individual values within an array cannot.

(admonition:immutable-arrays)=
:::{admonition} Advanced Users
An {class}`ak.Array` doesn't itself contain any data; it wraps a low-level {class}`ak.contents.Content` object that defines the structure of the array. Assigning to a field just replaces the existing {class}`ak.contents.Content` with a new {class}`ak.contents.Content`. Therefore, the {class}`ak.contents.Content` objects are immutable, whilst {class}`ak.Array` is not.
:::

+++

Using this syntax, we can assign to a _new_ field of an array:

```{code-cell} ipython3
array["y"] = [9, 8, 7]
array.show()
```

If necessary, the new field will be broadcasted to fit the array. For example, we can introduce a third field `z` that is set to the constant `0`:

```{code-cell} ipython3
array["z"] = 0
array.show()
```

A field can also be assigned deeply into a nested record e.g.

```{code-cell} ipython3
nested = ak.zip({"a": ak.zip({"x": [1, 2, 3]})})
nested["a", "y"] = 2 * nested.a.x

nested.show()
```

Note that the following does **not** work:

```{code-cell} ipython3
nested["a"]["y"] = 2 * nested.a.x  # does not work, nested["a"] is a copy!
nested.show()
```

Why does this happen? Well, Python first evaluates `nested["a"]`, which returns a _new_ {class}`ak.Array` that is a (shallow) copy of the data in `nested.a`. Hence, the next step — to set `y` — operates on a _different_  {class}`ak.Array`, and `nested.a` remains unchanged. The {ref}`Advanced Users<admonition:immutable-arrays>` call-out provides a more detailed explanation for _why_ this does not work.

+++

### Using `ak.with_field`

+++

Sometimes you might not want to modify an existing array, but rather produce a new array with the new field. Whilst this can be done using a shallow copy, e.g.

```{code-cell} ipython3
import copy

copied = copy.copy(nested)
copied["z"] = [10, 20, 30]

copied.show()
```

```{code-cell} ipython3
nested.show()
```

Awkward provides a dedicated function {func}`ak.with_field` that does this. 

:::{note}
Setting a field with `array['x']` uses {func}`ak.with_field` under the hood, so performance is not a factor in choosing one over the other.
:::
