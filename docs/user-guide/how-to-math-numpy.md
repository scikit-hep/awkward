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

How to perform computations with NumPy
======================================

Awkward Array's integration with NumPy allows you to use NumPy's array functions on data with complex structures, including ragged and heterogeneous arrays. 

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## Universal functions (ufuncs)

[NumPy's universal functions (ufuncs)](https://numpy.org/doc/stable/reference/ufuncs.html) are functions that operate elementwise on arrays. They are broadcasting-aware, so they can naturally handle data structures like ragged arrays that are common in Awkward Arrays.

Here's an example of applying `np.sqrt`, a NumPy ufunc, to an Awkward Array:

```{code-cell} ipython3
data = ak.Array([[1, 4, 9], [], [16, 25]])

np.sqrt(data)
```

Notice that the ufunc applies to the numeric data, passing through all dimensions of nested lists, even if those lists have variable length. This also applies to heterogeneous data, in which the data are not all of the same type.

```{code-cell} ipython3
data = ak.Array([[1, 4, 9], [], 16, [[[25]]]])

np.sqrt(data)
```

Unary and binary operations on Awkward Arrays, such as `+`, `-`, `>`, and `==`, are actually calling NumPy ufuncs. For instance, `+`:

```{code-cell} ipython3
array1 = ak.Array([[1, 2, 3], [], [4, 5]])
array2 = ak.Array([[10, 20, 30], [], [40, 50]])

array1 + array2
```

is actually `np.add`:

```{code-cell} ipython3
np.add(array1, array2)
```

### Arrays with record fields

Ufuncs can only be applied to numerical data in lists, not records.

```{code-cell} ipython3
records = ak.Array([{"x": 4, "y": 9}, {"x": 16, "y": 25}])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
np.sqrt(records)
```

However, you can pull each field out of a record and apply the ufunc to it.

```{code-cell} ipython3
np.sqrt(records.x)
```

```{code-cell} ipython3
np.sqrt(records.y)
```

If you want the result wrapped up in a new array of records, you can use {func}`ak.zip` to do that.

```{code-cell} ipython3
ak.zip({"x": np.sqrt(records.x), "y": np.sqrt(records.y)})
```

Here's an idiom that would apply a ufunc to every field individually, and then wrap up the result as a new record with the same fields (using {func}`ak.fields`, {func}`ak.unzip`, and {func}`ak.zip`):

```{code-cell} ipython3
ak.zip({key: np.sqrt(value) for key, value in zip(ak.fields(records), ak.unzip(records))})
```

The reaons that Awkward Array does not do this automatically is to prevent mistakes: it's common for records to represent coordinates of data points, and if the coordinates are not Cartesian, the one-to-one application is not correct.

+++

### Using non-NumPy ufuncs

NumPy-compatible ufuncs exist in other libraries, like SciPy, and can be applied in the same way. Here’s how you can apply `scipy.special.gamma` and `scipy.special.erf`:

```{code-cell} ipython3
import scipy.special

data = ak.Array([[0.1, 0.2, 0.3], [], [0.4, 0.5]])
```

```{code-cell} ipython3
scipy.special.gamma(data)
```

```{code-cell} ipython3
scipy.special.erf(data)
```

You can even create your own ufuncs using Numba's `@nb.vectorize`:

```{code-cell} ipython3
import numba as nb

@nb.vectorize
def gcd_euclid(x, y):
    # computation that is more complex than a formula
    while y != 0:
        x, y = y, x % y
    return x
```

```{code-cell} ipython3
x = ak.Array([[10, 20, 30], [], [40, 50]])
y = ak.Array([[5, 40, 15], [], [24, 255]])
```

```{code-cell} ipython3
gcd_euclid(x, y)
```

Since Numba has JIT-compiled this function, it would run much faster on large arrays than custom Python code.

+++

## Non-ufunc NumPy functions

Some NumPy functions don't satisfy the ufunc protocol, but have been implemented for Awkward Arrays because they are useful. You can tell when a NumPy function has an Awkward Array implementation when a function with the same name and signature exists in both libraries.

For instance, `np.where` works on Awkward Arrays because {func}`ak.where` exists:

```{code-cell} ipython3
np.where(y % 2 == 0, x, y) 
```

(The above selects elements from `x` when `y` is even and elements from `y` when `y` is odd.)

Similarly, `np.concatenate` works on Awkward Arrays because {func}`ak.concatenate` exists:

```{code-cell} ipython3
np.concatenate([x, y])
```

```{code-cell} ipython3
np.concatenate([x, y], axis=1)
```

Other NumPy functions, without an equivalent in the Awkward Array library, will work only if the Awkward Array can be converted into a NumPy array.

Ragged arrays can't be converted to NumPy:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [raises-exception]
---
np.fft.fft(ak.Array([[1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]))
```

But arrays with equal-sized lists can:

```{code-cell} ipython3
np.fft.fft(ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
```
