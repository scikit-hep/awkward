---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-input, remove-output]

%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"

import numpy as np
import awkward._v2 as ak
```

+++ {"tags": []}

# What is an Awkward Array?

+++

## Ragged (jagged) arrays

+++ {"tags": []}

Awkward Arrays are general tree-like data structures, like JSON, but contiguous in memory and operated upon with compiled, vectorized code like NumPy.

+++

They look like lists:

```{code-cell} ipython3
ak.Array([1, 2, 3])
```

They can contain sub-lists:

```{code-cell} ipython3
ak.Array([
    [1, 2, 3],
    [4, 5, 6]
])
```

These lists can have different lengths; arrays can be [jagged](https://en.wikipedia.org/wiki/Jagged_array):

```{code-cell} ipython3
ak.Array([
    [1, 2, 3],
    [4],
    [5, 6]
])
```

Each list can contain missing values:

```{code-cell} ipython3
ak.Array([
    [1, 2, 3],
    [4],
    [5, 6, None]
])
```

Awkward Arrays can store _numbers_:

```{code-cell} ipython3
digit_of_pi = ak.Array([
    [3, 141], 
    [59, 26, 53], 
    [58]
])
```

They can also work with _dates_:

```{code-cell} ipython3
ak.Array([
    [np.datetime64("1815-12-10"), np.datetime64("1969-07-16")], 
    [np.datetime64("1564-04-26")]
])
```

They can even work with _strings_:

```{code-cell} ipython3
ak.Array(
    [
        [
            "Benjamin List", "David MacMillan",            
        ],
        [
            "Emmanuelle Charpentier", "Jennifer A. Doudna",
        ],
    ]
)
```

Awkward Arrays can have structure through _records_:

```{code-cell} ipython3
ak.Array([
    [
        {"firstname":"Benjamin","surname":"List"},
        {"firstname":"David","surname":"MacMillan"}
    ],
    [
        {"firstname":"Emmanuelle","surname":"Charpentier"},
        {"firstname":"Jennifer A.","surname":"Doudna"}
    ],
])
```

In fact, Awkward Arrays can represent many kinds of jagged data. They can possess complex structures that mix lists, records, and primitive types. They can even contain _missing_ data and unions!

```{code-cell} ipython3
taxi_trip = ak.from_parquet(
    "https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet",
    row_groups=[0]
)
taxi_trip.type.show()
```

## NumPy-like API

Awkward Array _looks like_ NumPy:

```{code-cell} ipython3
ak.mean(digit_of_pi)
```

But generalises to the tricky kinds of data that NumPy struggles to work with:

```{code-cell} ipython3
ak.mean(digit_of_pi, axis=0)
```

NumPy can be coerced to dealing with jagged arrays, but the ensuing code is often more complex, memory-hungry, and verbose:

```{code-cell} ipython3
np.mean(
    np.ma.masked_invalid(
        [
            [3, 141, np.nan],
            [59, 26, 53],
            [58, np.nan, np.nan]
        ]
    ),
    axis=0
)
```

## Lightweight Structures

+++

Awkward makes it east to pull apart complex data structures:

```{code-cell} ipython3
taxi_trip.payment.fare
```

```{code-cell} ipython3
taxi_trip.payment.fare + taxi_trip.payment.tips
```

Its records are lightweight and simple to compose:

```{code-cell} ipython3
ak.zip({
    "fare": taxi_trip.payment.fare,
    "total": taxi_trip.payment.fare + taxi_trip.payment.tips
})
```

## High performance

+++

Like NumPy, Awkward Array performs computations in fast, optimised kernels:

```{code-cell} ipython3
%%timeit latdiff = taxi_trip.trip.path.latdiff

ak.sum(latdiff)
```

Look how much slower a pure-Python sum over the flattened array is:

```{code-cell} ipython3
%%timeit latdiff = ak.ravel(taxi_trip.trip.path.latdiff)

sum(latdiff)
```
