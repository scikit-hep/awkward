---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.10'
    jupytext_version: 1.5.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to convert to/from NumPy
============================

As a generalization of NumPy, any NumPy array can be converted to an Awkward Array, but not vice-versa.

```{code-cell} ipython3
import awkward1 as ak
import numpy as np
```

The general function for NumPy → Awkward is [ak.from_numpy](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_numpy.html).

```{code-cell} ipython3
np_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
np_array
```

```{code-cell} ipython3
ak_array = ak.from_numpy(np_array)
ak_array
```
