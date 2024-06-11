---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What is an "Awkward" Array?


```{code-cell} ipython3
import numpy as np
import awkward as ak
```

## Versatile Arrays
Awkward Arrays are general tree-like data structures, like JSON, but contiguous in memory and operated upon with compiled, vectorized code like NumPy.

They look like NumPy arrays:


```{code-cell} ipython3
ak.Array([1, 2, 3])
```

Like NumPy, they can have multiple dimensions:


```{code-cell} ipython3
ak.Array([
    [1, 2, 3],
    [4, 5, 6]
])
```

These dimensions can have varying lengths; arrays can be [ragged](https://en.wikipedia.org/wiki/Jagged_array):


```{code-cell} ipython3
ak.Array([
    [1, 2, 3],
    [4],
    [5, 6]
])
```

Each dimension can contain missing values:


```{code-cell} ipython3
ak.Array([
    [1, 2, 3],
    [4],
    [5, 6, None]
])
```

Awkward Arrays can store _numbers_:


```{code-cell} ipython3
ak.Array([
    [3, 141], 
    [59, 26, 535], 
    [8]
])
```

They can also work with _dates_:


```{code-cell} ipython3
ak.Array(
    [
        [np.datetime64("1815-12-10"), np.datetime64("1969-07-16")],
        [np.datetime64("1564-04-26")],
    ]
)
```

They can even work with _strings_:


```{code-cell} ipython3
ak.Array(
    [
        [
            "Benjamin List",
            "David MacMillan",
        ],
        [
            "Emmanuelle Charpentier",
            "Jennifer A. Doudna",
        ],
    ]
)
```

Awkward Arrays can have structure through _records_:


```{code-cell} ipython3
ak.Array(
    [
        [
            {"name": "Benjamin List", "age": 53},
            {"name": "David MacMillan", "age": 53},
        ],
        [
            {"name": "Emmanuelle Charpentier", "age": 52},
            {"name": "Jennifer A. Doudna", "age": 57},
        ],
        [
            {"name": "Akira Yoshino", "age": 73},
            {"name": "M. Stanley Whittingham", "age": 79},
            {"name": "John B. Goodenough", "age": 98},
        ],
    ]
)
```

In fact, Awkward Arrays can represent many kinds of jagged data. They can possess complex structures that mix records, and primitive types.


```{code-cell} ipython3
ak.Array(
    [
        [
            {
                "name": "Benjamin List",
                "age": 53,
                "institutions": [
                    "University of Cologne",
                    "Max Planck Institute for Coal Research",
                    "Hokkaido University",
                ],
            },
            {
                "name": "David MacMillan",
                "age": 53,
                "institutions": None,
            },
        ]
    ]
)
```

They can even contain unions!


```{code-cell} ipython3
ak.Array(
    [
        [np.datetime64("1815-12-10"), "Cassini"],
        [np.datetime64("1564-04-26")],
    ]
)
```

## NumPy-like interface

Awkward Array _looks like_ NumPy. It behaves identically to NumPy for regular arrays


```{code-cell} ipython3
x = ak.Array([
    [1, 2, 3],
    [4, 5, 6]
]);
```


```{code-cell} ipython3
ak.sum(x, axis=-1)
```

providing a similar high-level API, and implementing the [ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) mechanism:


```{code-cell} ipython3
powers_of_two = ak.Array(
    [
        [1, 2, 4],
        [None, 8],
        [16],
    ]
);
```


```{code-cell} ipython3
ak.sum(powers_of_two)
```

But generalises to the tricky kinds of data that NumPy struggles to work with. It can perform reductions through varying length lists:

![](example-reduction-sum.svg)


```{code-cell} ipython3
ak.sum(powers_of_two, axis=0)
```

## Lightweight structures
Awkward makes it east to pull apart record structures:


```{code-cell} ipython3
nobel_prize_winner = ak.Array(
    [
        [
            {"name": "Benjamin List", "age": 53},
            {"name": "David MacMillan", "age": 53},
        ],
        [
            {"name": "Emmanuelle Charpentier", "age": 52},
            {"name": "Jennifer A. Doudna", "age": 57},
        ],
        [
            {"name": "Akira Yoshino", "age": 73},
            {"name": "M. Stanley Whittingham", "age": 79},
            {"name": "John B. Goodenough", "age": 98},
        ],
    ]
);
```


```{code-cell} ipython3
nobel_prize_winner.name
```


```{code-cell} ipython3
nobel_prize_winner.age
```

These records are lightweight, and simple to compose:


```{code-cell} ipython3
nobel_prize_winner_with_birth_year = ak.zip({
    "name": nobel_prize_winner.name,
    "age": nobel_prize_winner.age,
    "birth_year": 2021 - nobel_prize_winner.age
});
```


```{code-cell} ipython3
nobel_prize_winner_with_birth_year.show()
```

## High performance
Like NumPy, Awkward Array performs computations in fast, optimised kernels.


```{code-cell} ipython3
large_array = ak.Array([[1, 2, 3], [], [4, 5]] * 1_000_000)
```

We can compute the sum in `3.37 ms ± 107 µs` on a reference CPU:


```{code-cell} ipython3
ak.sum(large_array)
```

The same sum can be computed with pure-Python over the flattened array in `369 ms ± 8.07 ms`:


```{code-cell} ipython3
large_flat_array = ak.ravel(large_array)

sum(large_flat_array)
```

These performance values are not benchmarks; they are only an indication of the speed of Awkward Array.

Some problems are hard to solve with array-oriented programming. Awkward Array supports [Numba](https://numba.pydata.org/) out of the box:

```{code-cell} ipython3
import numba as nb

@nb.njit
def cumulative_sum(arr):
    result = 0
    for x in arr:
        for y in x:
            result += y
    return result
    
cumulative_sum(large_array)
```
