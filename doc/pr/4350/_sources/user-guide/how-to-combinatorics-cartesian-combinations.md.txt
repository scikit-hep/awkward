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

How to find all combinations of elements: Cartesian (cross) product and "n choose k"
====================================================================================

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

## Motivation

In non-array code that operates on arbitrary data structures, such as Python for loops and Python objects, doubly nested for loops like the following are pretty common:

```{code-cell} ipython3
class City:
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

cities_us = [
    City("New York", 40.7128, -74.0060),
    City("Los Angeles", 34.0522, -118.2437),
    City("Chicago", 41.8781, -87.6298),
]
cities_canada = [
    City("Toronto", 43.6510, -79.3470),
    City("Vancouver", 49.2827, -123.1207),
    City("Montreal", 45.5017, -73.5673),
]
```

Cartesian product:

```{code-cell} ipython3
class CityPair:
    def __init__(self, city1, city2):
        self.city1 = city1
        self.city2 = city2
    def __repr__(self):
        return f"<CityPair {self.city1.name} {self.city2.name}>"

pairs = []

for city_us in cities_us:
    for city_canada in cities_canada:
        pairs.append(CityPair(city_us, city_canada))

pairs
```

and "n choose k" (combinations without replacement):

```{code-cell} ipython3
all_cities = cities_us + cities_canada
```

```{code-cell} ipython3
pairs = []

for i, city1 in enumerate(all_cities):
    for city2 in all_cities[i + 1:]:
        pairs.append(CityPair(city1, city2))

pairs
```

These kinds of combinations are common enough that there are special functions for them in Python's [itertools](https://docs.python.org/3/library/itertools.html) library:

* [itertools.product](https://docs.python.org/3/library/itertools.html#itertools.product) for the Cartesian product,
* [itertools.combinations](https://docs.python.org/3/library/itertools.html#itertools.combinations) for combinations without replacement.

```{code-cell} ipython3
import itertools
```

```{code-cell} ipython3
list(
    CityPair(city1, city2)
    for city1, city2 in itertools.product(cities_us, cities_canada)
)
```

```{code-cell} ipython3
list(
    CityPair(city1, city2)
    for city1, city2 in itertools.combinations(all_cities, 2)
)
```

Awkward Array has special functions for these kinds of combinations as well:

* {func}`ak.cartesian` for the Cartesian product,
* {func}`ak.combinations` for combinations without replacement.

```{code-cell} ipython3
def instance_to_dict(city):
    return {"name": city.name, "latitude": city.latitude, "longitude": city.longitude}

cities_us = ak.Array([instance_to_dict(city) for city in cities_us])
cities_canada = ak.Array([instance_to_dict(city) for city in cities_canada])

all_cities = ak.concatenate([cities_us, cities_canada])
```

```{code-cell} ipython3
ak.cartesian([cities_us, cities_canada], axis=0)
```

```{code-cell} ipython3
ak.combinations(all_cities, 2, axis=0)
```

## Combinations with `axis=1`

The default `axis` for these functions is 1, rather than 0, as in the motivating example. Problems that are big enough to benefit from vectorized combinations would produce very large output arrays, which likely wouldn't fit in any computer's memory. (Those problems are a better fit for SQL's `CROSS JOIN`; note that Python has a built-in interface to [sqlite3](https://docs.python.org/3/library/sqlite3.html) in-memory tables. You could even use SQL to populate an array of integer indexes to later slice an Awkward Array...)

The most useful application of Awkward Array combinatorics are on problems in which small, variable-length lists need to be combined—and there are many of them. This is `axis=1` (default) or `axis > 1`.

Here is an example of many Cartesian products:

![](cartoon-cartesian.png)

```{code-cell} ipython3
numbers = ak.Array([[1, 2, 3], [], [4, 5], [6, 7, 8, 9]] * 250)
letters = ak.Array([["a", "b"], ["c"], ["d", "e", "f", "g"], ["h", "i"]] * 250)
```

```{code-cell} ipython3
ak.cartesian([numbers, letters])
```

Here is an example of many combinations without replacement:

![](cartoon-combinations.png)

```{code-cell} ipython3
ak.combinations(numbers, 2)
```

## Calculations on pairs

Usually, you'll want to do some calculation on each pair (or on each triple or quadruple, etc.). To get the left-side and right-side of each pair into separate arrays, so they can be used in a calculation, you could address the members of the tuple individually:

```{code-cell} ipython3
tuples = ak.combinations(numbers, 2)
```

```{code-cell} ipython3
tuples["0"], tuples["1"]
```

Be sure to use integers in strings when addressing fields of a tuple ("columns") and plain integers when addressing array elements ("rows"). The above is different from

```{code-cell} ipython3
tuples[0], tuples[1]
```

Once they're in separate arrays, they can be used in a formula:

```{code-cell} ipython3
tuples["0"] * tuples["1"]
```

Another way to get fields of a tuple (or fields of a record) as individual arrays is to use {func}`ak.unzip`:

```{code-cell} ipython3
lefts, rights = ak.unzip(tuples)

lefts * rights
```

## Maintaining groups

In combinations like

```{code-cell} ipython3
ak.cartesian([np.arange(5), np.arange(4)], axis=0)
```

produce a flat list of combinations, but some calculations need triples with the same first or second value in the same list, for instance if they're going to {func}`ak.max` over lists ("find the best combination in which...") or compute {func}`ak.any` or {func}`ak.all` ("is there any combination in which...?"). The `nested` argument controls this.

```{code-cell} ipython3
result = ak.cartesian([np.arange(5), np.arange(4)], axis=0, nested=True)
result
```

For instance, "is there any combination in which |_left_ - _right_| ≥ 3?"

```{code-cell} ipython3
lefts, rights = ak.unzip(result)

ak.any(abs(lefts - rights) >= 3, axis=1)
```
