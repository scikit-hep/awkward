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

How to create arrays of missing data
====================================

Data at any level of an Awkward Array can be "missing," represented by `None` in Python.

Some of this functionality is like NumPy's [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.html), but NumPy masked arrays can only declare numerical values to be missing (not, for instance, a row of a 2-dimensional array) and they represent missing data with `np.ma.masked` instead of `None`.

Pandas users are accustomed to using floating-point `NaN` (not a number) values for missing data, though Pandas has introduced a [first-class missing value](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#missing-data-na) (`pd.NA`) in version 1.0. Awkward Array's approach is like this, but data in an Awkward Array are never Python objects, so there's no specialized Python object representing it, other than `None` when all the data are converted from or to Python with [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) and [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html).

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

Missing data are not `NaN`!

```{code-cell} ipython3

```

```{code-cell} ipython3

```
