---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to create arrays with ArrayBuilder (easy and general)
=========================================================

If you're not getting data from a file or conversion from another format, you may need to create it from scratch. [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) is a general, high-level way to do that, though it has performance limitations.

The biggest difference between an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) and an [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) is that you can append data to a builder, but an array is immutable ([see qualifications on mutability](how-to-convert-numpy.html#mutability-of-awkward-arrays-from-numpy)). It's a bit like a Python list, which has an `append` method, but [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) has many methods for appending different types of structures.

```{code-cell} ipython3
import awkward as ak
```

When a builder is first created, it has zero length and unknown type.

```{code-cell} ipython3
builder = ak.ArrayBuilder()
builder
```

Calling its `append` method adds data and also determines its type.

```{code-cell} ipython3
builder.append(1)
builder
```

```{code-cell} ipython3
builder.append(2.2)
builder
```

```{code-cell} ipython3
builder.append(3+1j)
builder
```

Note that this can include missing data by promoting to an [option-type](https://awkward-array.readthedocs.io/en/latest/ak.types.OptionType.html),

```{code-cell} ipython3
builder.append(None)
builder
```

and mix types by promoting to a [union-type](https://awkward-array.readthedocs.io/en/latest/ak.types.UnionType.html):

```{code-cell} ipython3
builder.append("five")
builder
```

```{code-cell} ipython3
ak.type(builder)
```

We've been using "`append`" because it is generic (it recognizes the types of its arguments and builds that), but there are also methods for building structure explicitly.

```{code-cell} ipython3
builder = ak.ArrayBuilder()
builder.boolean(False)
builder.integer(1)
builder.real(2.2)
builder.complex(3+1j)
builder.null()
builder.string("five")
builder
```

```{code-cell} ipython3

```
