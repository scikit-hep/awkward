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

How to convert to/from NumPy
============================

As a generalization of NumPy, any NumPy array can be converted to an Awkward Array, but not vice-versa.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

From NumPy to Awkward
---------------------

The function for NumPy → Awkward conversion is {func}`ak.from_numpy`.

```{code-cell} ipython3
np_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
np_array
```

```{code-cell} ipython3
ak_array = ak.from_numpy(np_array)
ak_array
```

However, NumPy arrays are also recognized by the {class}`ak.Array` constructor, so you can use that unless your goal is to explicitly draw the reader's attention to the fact that the input is a NumPy array.

```{code-cell} ipython3
ak_array = ak.Array(np_array)
ak_array
```

Fixed-size vs variable-length dimensions
----------------------------------------

If the NumPy array is multidimensional, the Awkward Array will be as well.

```{code-cell} ipython3
np_array = np.array([[100, 200], [101, 201], [103, 203]])
np_array
```

```{code-cell} ipython3
ak_array = ak.Array(np_array)
ak_array
```

It's important to notice that the type is `3 * 2 * int64`, not `3 * var * int64`. The second dimension has a fixed size—it is guaranteed to have exactly two items—just like a NumPy array. This differs from an Awkward Array constructed from Python lists:

```{code-cell} ipython3
ak.Array([[100, 200], [101, 201], [103, 203]])
```

or JSON:

```{code-cell} ipython3
ak.Array("[[100, 200], [101, 201], [103, 203]]")
```

because Python and JSON lists have arbitrary lengths, at least in principle, if not in a particular instance. Some behaviors depend on this fact (such as broadcasting rules).

+++

From Awkward to NumPy
---------------------

The function for Awkward → NumPy conversion is {func}`ak.to_numpy`.

```{code-cell} ipython3
np_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
ak_array = ak.Array(np_array)
ak_array
```

```{code-cell} ipython3
ak.to_numpy(ak_array)
```

Awkward Arrays that happen to have regular structure can be converted to NumPy, even if their type is formally "variable length lists" (`var`):

```{code-cell} ipython3
ak_array = ak.Array([[1, 2, 3], [4, 5, 6]])
ak_array
```

```{code-cell} ipython3
ak.to_numpy(ak_array)
```

But if the lengths of nested lists do vary, attempts to convert to NumPy fail:

```{code-cell} ipython3
ak_array = ak.Array([[1, 2, 3], [], [4, 5]])
ak_array
```

```{code-cell} ipython3
:tags: [raises-exception]

ak.to_numpy(ak_array)
```

One might argue that such arrays should become NumPy arrays with `dtype="O"`. However, this is usually undesirable because these "NumPy object arrays" are just arrays of pointers to Python objects, and all the performance issues of dealing with Python objects apply.

If you do want this, use {func}`ak.to_list` with the {class}`np.ndarray` constructor.

```{code-cell} ipython3
np.array(ak.to_list(ak_array), dtype="O")
```

Implicit Awkward to NumPy conversion
------------------------------------

Awkward Arrays satisfy NumPy's `__array__` protocol, so simply passing an Awkward Array to the {class}`np.ndarray` constructor calls {func}`ak.to_numpy`.

```{code-cell} ipython3
ak_array = ak.Array([[1, 2, 3], [4, 5, 6]])
ak_array
```

```{code-cell} ipython3
np.array(ak_array)
```

Libraries that expect NumPy arrays as input, such as Matplotlib, use this.

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.plot(ak_array);
```

Implicit conversion to NumPy inherits the same restrictions as {func}`ak.to_numpy`, namely that variable-length lists cannot be converted to NumPy.

```{code-cell} ipython3
ak_array = ak.Array([[1, 2, 3], [], [4, 5]])
ak_array
```

```{code-cell} ipython3
:tags: [raises-exception]

np.array(ak_array)
```

NumPy's structured arrays
-------------------------

[NumPy's structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) correspond to Awkward's "record type."

```{code-cell} ipython3
np_array = np.array(
    [(1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4), (5, 5.5)], dtype=[("x", int), ("y", float)]
)
np_array
```

```{code-cell} ipython3
ak_array = ak.from_numpy(np_array)
ak_array
```

```{code-cell} ipython3
ak.to_numpy(ak_array)
```

Awkward Arrays with record type can be sliced by field name like NumPy structured arrays:

```{code-cell} ipython3
ak_array["x"]
```

```{code-cell} ipython3
np_array["x"]
```

But Awkward Arrays can be sliced by field name _and_ index within the same square brackets, whereas NumPy requires two sets of square brackets.

```{code-cell} ipython3
ak_array["x", 2]
```

```{code-cell} ipython3
:tags: [raises-exception]

np_array["x", 2]
```

```{code-cell} ipython3
np_array["x"][2]
```

They have the same commutivity, however. In this example, slicing `"x"` and then `2` returns the same result as `2` and then `"x"`.

```{code-cell} ipython3
ak_array[2, "x"]
```

```{code-cell} ipython3
np_array[2]["x"]
```

NumPy's masked arrays
---------------------

[NumPy's masked arrays](https://numpy.org/doc/stable/reference/maskedarray.generic.html) correspond to Awkward's "option type."

```{code-cell} ipython3
np_array = np.ma.MaskedArray(
    [[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [True, True, False]]
)
np_array
```

```{code-cell} ipython3
np_array.tolist()
```

```{code-cell} ipython3
ak_array = ak.from_numpy(np_array)
ak_array
```

The `?` before `int64` (expands to `option[...]` for more complex contents) refers to "option type," meaning that the values can be missing ("None" in Python).

It is possible for a dataset to have no missing data, yet still have option type, just as it's possible to have a NumPy masked array with no mask.

```{code-cell} ipython3
ak.from_numpy(np.ma.MaskedArray([[1, 2, 3], [4, 5, 6]], mask=False))
```

Awkward Arrays with option type are converted to NumPy masked arrays.

```{code-cell} ipython3
ak.to_numpy(ak_array)
```

```{code-cell} ipython3
ak.to_numpy(ak_array).tolist()
```

Note, however, that the structure of an Awkward Array's option type is not always preserved when converting to NumPy masked arrays. Masked arrays can only have missing numbers, not missing lists, so missing lists are expanded into lists of missing numbers.

For example, an array of type `var * ?int64` can be converted into an identical NumPy structure:

```{code-cell} ipython3
ak_array1 = ak.Array([[1, None, 3], [None, None, 6]])
ak_array1
```

```{code-cell} ipython3
ak.to_numpy(ak_array1).tolist()
```

But an array of type `option[var * int64]` must have its missing lists expanded into lists of missing numbers.

```{code-cell} ipython3
ak_array2 = ak.Array([[1, 2, 3], None, [4, 5, 6]])
ak_array2
```

```{code-cell} ipython3
ak.to_numpy(ak_array2).tolist()
```

Finally, it is possible to prevent the {func}`ak.to_numpy` function from creating NumPy masked arrays by passing `allow_missing=False`.

```{code-cell} ipython3
:tags: [raises-exception]

ak.to_numpy(ak_array, allow_missing=False)
```

You might want to do this to be sure that the output of {func}`ak.to_numpy` has type {class}`np.ndarray` (or die trying).

+++

NumpyArray shapes vs RegularArrays
----------------------------------

```{note}
Advanced topic: it is not necessary to understand the internal representation in order to use Awkward Arrays in data analysis.
```

One reason you might want to use {func}`ak.from_numpy` directly is to control how it is internally represented.

Inside of an {class}`ak.Array`, data structures are represented by "layout nodes" such as {class}`ak.contents.NumpyArray` and {class}`ak.contents.RegularArray`.

```{code-cell} ipython3
np_array = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype="i1")
ak_array1 = ak.from_numpy(np_array)
ak_array1.layout
```

In the above, the shape is represented as part of the {class}`ak.contents.NumpyArray` node, but it could also have been represented in {class}`ak.contents.RegularArray` nodes.

```{code-cell} ipython3
ak_array2 = ak.from_numpy(np_array, regulararray=True)
ak_array2.layout
```

In the above, the internal {class}`ak.contents.NumpyArray` is one-dimensional and the shape is described by nesting it within two {class}`ak.contents.RegularArray` nodes.

This distinction is technical: `ak_array1` and `ak_array2` have the same {func}`ak.type` and behave identically (including broadcasting rules).

```{code-cell} ipython3
ak.type(ak_array1)
```

```{code-cell} ipython3
ak.type(ak_array2)
```

```{code-cell} ipython3
ak_array1 == ak_array2
```

```{code-cell} ipython3
ak.all(ak_array1 == ak_array2)
```

Mutability of Awkward Arrays from NumPy
---------------------------------------

```{note}
Advanced topic: unless you're willing to investigate subtleties of when a NumPy array is viewed and when it is copied, do not modify the NumPy arrays that Awkward Arrays are built from (or build Awkward Arrays from deliberate copies of the NumPy arrays).
```

Awkward Arrays are not supposed to be changed in place ("mutated"), and all of the functions in the Awkward Array library return new values, rather than changing the old. However, it is possible to create an Awkward Array from a NumPy array and modify the NumPy array in place, thus modifying the Awkward Array. Wherever possible, Awkward Arrays are _views_ of the NumPy data, not _copies_.

```{code-cell} ipython3
np_array = np.array([[1, 2, 3], [4, 5, 6]])
np_array
```

```{code-cell} ipython3
ak_array = ak.from_numpy(np_array)
ak_array
```

```{code-cell} ipython3
# Change the NumPy array in place.
np_array *= 100
np_array
```

```{code-cell} ipython3
# The Awkward Array changes as well.
ak_array
```

You might want to do this in some performance-critical applications. However, note that NumPy arrays sometimes have to be copied to make an Awkward Array.

For example, if a NumPy array is not C-contiguous and is internally represented as a {class}`ak.contents.RegularArray` (see previous section), it must be copied.

```{code-cell} ipython3
# Slicing the inner dimension of this NumPy array makes it not C-contiguous.
np_array = np.array([[1, 2, 3], [4, 5, 6]])
np_array.flags["C_CONTIGUOUS"], np_array[:, :-1].flags["C_CONTIGUOUS"]
```

```{code-cell} ipython3
# Case 1: C-contiguous and not RegularArray (should view).
ak_array1 = ak.from_numpy(np_array)
ak_array1
```

```{code-cell} ipython3
# Case 2: C-contiguous and RegularArray (should view).
ak_array2 = ak.from_numpy(np_array, regulararray=True)
ak_array2
```

```{code-cell} ipython3
# Case 3: not C-contiguous and not RegularArray (should view).
ak_array3 = ak.from_numpy(np_array[:, :-1])
ak_array3
```

```{code-cell} ipython3
# Case 4: not C-contiguous and RegularArray (has to copy).
ak_array4 = ak.from_numpy(np_array[:, :-1], regulararray=True)
ak_array4
```

```{code-cell} ipython3
# Change the NumPy array in place.
np_array *= 100
np_array[:, :-1]
```

```{code-cell} ipython3
# Case 1 changes as well because it is a view.
ak_array1
```

```{code-cell} ipython3
# Case 2 changes as well because it is a view.
ak_array2
```

```{code-cell} ipython3
# Case 3 changes as well because it is a view.
ak_array3
```

```{code-cell} ipython3
# Case 4 does not change because it is a copy.
ak_array4
```

In general, it can be hard to determine if an Awkward Array is a view or a copy because some operations need to construct a {class}`ak.contents.RegularArray`. Furthermore, the view-vs-copy behavior can change from one version of Awkward Array to the next. It is only safe to rely on view-vs-copy behavior of Awkward Arrays that were directly created from NumPy arrays, as in the four cases above, not in any derived arrays (i.e. arrays produced from slices of Awkward Arrays or computed using functions from the Awkward Array library).

+++

Mutability of Awkward Arrays converted to NumPy
-----------------------------------------------

```{note}
Advanced topic: unless you're willing to investigate subtleties of when an Awkward array is viewed and when it is copied, do not modify the NumPy arrays that Awkward Arrays are converted into (or make deliberate copies of the resulting NumPy arrays).
```

The considerations described above also apply to NumPy arrays created from Awkward Arrays. If possible, they are _views_, rather than _copies_, but these semantics are not guaranteed.

```{code-cell} ipython3
ak_array = ak.Array([[1, 2, 3], [4, 5, 6]])
ak_array
```

```{code-cell} ipython3
np_array = ak.to_numpy(ak_array)
np_array
```

```{code-cell} ipython3
# Change the NumPy array in place.
np_array *= 100
np_array
```

```{code-cell} ipython3
# The Awkward Array that it came from is changed as well.
ak_array
```

As a counter-example, a NumPy array constructed from an Awkward Array with missing data _might not_ be a view. (It depends on the internal representation; the most common case of an {class}`ak.contents.IndexedOptionArray` is not.)

```{code-cell} ipython3
ak_array1 = ak.Array([[1, None, 3], [None, None, 6]])
ak_array1
```

```{code-cell} ipython3
np_array = ak.to_numpy(ak_array1)
np_array
```

```{code-cell} ipython3
# Change the NumPy array in place.
np_array *= 100
np_array
```

```{code-cell} ipython3
:tags: []

# The Awkward Array that it came from is not changed in this case.
ak_array1
```
