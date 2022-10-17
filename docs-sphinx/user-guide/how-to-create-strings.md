---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to create arrays of strings
===============================

Awkward Arrays can contain strings, although these strings are just a special view of lists of `uint8` numbers. As such, the variable-length data are efficiently stored.

NumPy's strings are padded to have equal width, and Pandas's strings are Python objects. Awkward Array doesn't have nearly as many functions for manipulating arrays of strings as NumPy and Pandas, though.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

From Python strings
-------------------

The {class}`ak.Array` constructor and {func}`ak.from_iter` recognize strings, and strings are returned by {func}`ak.to_list`.

```{code-cell} ipython3
ak.Array(["one", "two", "three"])
```

They may be nested within anything.

```{code-cell} ipython3
ak.Array([["one", "two"], [], ["three"]])
```

From NumPy arrays
-----------------

NumPy strings are also recognized by {func}`ak.from_numpy` and {func}`ak.to_numpy`.

```{code-cell} ipython3
numpy_array = np.array(["one", "two", "three", "four"])
numpy_array
```

```{code-cell} ipython3
awkward_array = ak.Array(numpy_array)
awkward_array
```

Operations with strings
-----------------------

Since strings are really just lists, some of the list operations "just work" on strings.

```{code-cell} ipython3
ak.num(awkward_array)
```

```{code-cell} ipython3
awkward_array[:, 1:]
```

Others had to be specially overloaded for the string case, such as string-equality. The default meaning for `==` would be to descend to the lowest level and compare numbers (characters, in this case).

```{code-cell} ipython3
awkward_array == "three"
```

```{code-cell} ipython3
awkward_array == ak.Array(["ONE", "TWO", "three", "four"])
```

Similarly, {func}`ak.sort` and {func}`ak.argsort` sort strings lexicographically, not individual characters.

```{code-cell} ipython3
ak.sort(awkward_array)
```

Still other operations had to be inhibited, since they wouldn't make sense for strings.

```{code-cell} ipython3
:tags: [raises-exception]

np.sqrt(awkward_array)
```

Categorical strings
-------------------

A large set of strings with few unique values are more efficiently manipulated as integers than as strings. In Pandas, this is [categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html), in R, it's called a [factor](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/factor), and in Arrow and Parquet, it's [dictionary encoding](https://arrow.apache.org/blog/2019/09/05/faster-strings-cpp-parquet/).

The {func}`ak.to_categorical` function makes Awkward Arrays categorical in this sense. {func}`ak.to_arrow` and {func}`ak.to_parquet` recognize categorical data and convert it to the corresponding Arrow and Parquet types.

```{code-cell} ipython3
uncategorized = ak.Array(["three", "one", "two", "two", "three", "one", "one", "one"])
uncategorized
```

```{code-cell} ipython3
categorized = ak.to_categorical(uncategorized)
categorized
```

Internally, the data now have an index that selects from a set of unique strings.

```{code-cell} ipython3
categorized.layout.index
```

```{code-cell} ipython3
ak.Array(categorized.layout.content)
```

The main advantage to Awkward categorical data (other than proper conversions to Arrow and Parquet) is that equality is performed using the index integers.

```{code-cell} ipython3
categorized == "one"
```

With ArrayBuilder
-----------------

{func}`ak.ArrayBuilder` is described in more detail [in this tutorial](how-to-create-arraybuilder), but you can add strings by calling the `string` method or simply appending them.

(This is what {func}`ak.from_iter` uses internally to accumulate data.)

```{code-cell} ipython3
builder = ak.ArrayBuilder()

builder.string("one")
builder.append("two")
builder.append("three")

array = builder.snapshot()
array
```
