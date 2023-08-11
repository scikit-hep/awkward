---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

How to create arrays by "unflattening" or "grouping"
====================================================

```{code-cell} ipython3
import awkward as ak
import pandas as pd
import numpy as np
from urllib.request import urlopen
```

## Finding runs in an array
It is often the case that one has an array of data that they wish to subdivide into common groups. Let's imagine that we're looking at NASA's [Earth Meteorite Landings dataset](https://data.nasa.gov/resource/y77d-th95.json), and that we wish to find the largest meteorite in each classification. This is known as a `groupby` operation, followed by a reduction.

+++

First, we should load the data

```{code-cell} ipython3
with urlopen("https://data.nasa.gov/resource/y77d-th95.json") as f:
    landing = ak.from_json(f)
landing.fields
```

In order to find the _largest_ meteorite by each category, we must first group the entries into categories. This is called a `groupby` operation, whereby we are ordering the entire array into subgroups given by a particular label. To perform a `groupby` in Awkward Array, we must first sort the array by the category

```{code-cell} ipython3
landing_sorted_class = landing[ak.argsort(landing.recclass)]
landing_sorted_class
```

This sorted array can be subdivided into sublists of the same category. To determine how long each of these sublists must be, Awkward provides _another_ function {func}`ak.run_lengths` which, as the name implies, finds the lengths of consecutive _runs_ in an array, e.g.

```{code-cell} ipython3
ak.run_lengths([1, 1, 1, 3, 3, 2, 4, 4, 4])
```

The function does not accept an `axis` argument; Awkward Array only supports finding runs in the innermost `axis=-1` axis of the array. Let's find the lengths of each category sublist using {func}`ak.run_lengths`:

```{code-cell} ipython3
lengths = ak.run_lengths(landing_sorted_class.recclass)
lengths
```

## Dividing an array into sublists

+++

Awkward Array provides an {func}`ak.unflatten` operation that adds a new dimension to an array, using either a single integer denoting the (regular) size of the dimension, or a list of integers representing the lengths of the sublists to create e.g.

```{code-cell} ipython3
ak.unflatten(
    ["Do", "re", "mi", "fa", "so", "la"],
    [1, 2, 2, 1]
)
```

If we pass an integer instead of a list of lengths, we get a regular array

```{code-cell} ipython3
ak.unflatten(
    ["Do", "re", "mi", "fa", "so", "la"],
    2
)
```

We can unflatten our sorted array using the length of runs each classification, in order to finalise our groupby operation.

```{code-cell} ipython3
landing_by_class = ak.unflatten(
    landing_sorted_class, 
    lengths
)
landing_by_class
```

We can see the categories of this grouped array by pulling out the first item of each sublist

```{code-cell} ipython3
landing_by_class.recclass[..., 0]
```

The above three steps:
1. Sort the array
2. Compute the length of runs within the sorted array
3. Unflatten the sorted array by the run lengths

form a `groupby` operation.

+++

### Computing the mass of the largest meteorites

+++

Now that we have grouped our meteorite landings by classification, we can find the largest mass meteorite in each group. If we look at the type of the array, we can see that the `mass` field is actually a string:

```{code-cell} ipython3
landing_by_class.type.show()
```

Let's convert it to a floating point number

```{code-cell} ipython3
landing_by_class['mass'] = ak.strings_astype(landing_by_class.mass, np.float64)
```

Now we can find the index of the largest mass in each sublist. We'll use `keepdims=True` in order to be able to use this array to index `landing_by_class` and pull out the corresponding record.

```{code-cell} ipython3
i_largest_mass = ak.argmax(landing_by_class.mass, axis=-1, keepdims=True)
```

Finding the largest meteorite is then a simple case of using `i_largest_mass` as an index, and flattening the result to drop the unneeded dimension

```{code-cell} ipython3
largest_meteorite = ak.flatten(
    landing_by_class[i_largest_mass], 
    axis=1,
)
largest_meteorite
```

Here are there names!

```{code-cell} ipython3
largest_meteorite.name
```
