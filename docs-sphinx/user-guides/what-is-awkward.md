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
:tags: [hide-cell]

%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"

import numpy as np
import ipyleaflet as ipl
import awkward._v2 as ak
```

+++ {"tags": []}

What is an Awkward Array?
=========================

Efficiency and generality
-------------------------

Arrays are the most efficient data structures for sequential numeric processing, and [NumPy](https://numpy.org/) makes it easy to interact with arrays in Python. However, NumPy's arrays are rectangular tables or tensors that cannot express variable-length structures. 

General tree-like data are often expressed using [JSON](https://www.json.org/), but at the expense of memory use and processing speed.

Awkward Arrays are general tree-like data structures, like JSON, but contiguous in memory and operated upon with compiled, vectorized code like NumPy. They're basic building blocks for data analyses that are, well, more awkward than those involving neat tables.

This library was originally developed for high-energy particle physics. Particle physics datasets have rich data structures that usually can't be flattened into rectangular arrays, but physicists need to process them efficiently because the datasets are enormous. Awkward Arrays combine generic data structures with high-performance number-crunching.

Let's illustrate this with a non-physics dataset: maps of taxi routes in the city of Chicago. 

The City of Chicago has a [Data Portal](https://data.cityofchicago.org/) with lots of interesting datasets. 
This exercise uses a dataset of [Chicago taxi trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) taken from 2019 through 2021 (3 years).

Here's a map of Chicago, for reference:

+++

:::{figure} https://upload.wikimedia.org/wikipedia/commons/3/3f/USA_Chicago_location_map.svg

Map of Chicago, with Lake Michigan shaded blue and the airport indicated as a yellow circle in the top-left quadrant.
:::

+++

The dataset that the Data Portal provides has trip start and stop points as longitude, latitude pairs, as well as start and end times (date-stamps), payment details, and the name of each taxi company. For this example, an estimated route of each taxi trip has been computed by [Open Source Routing Machine (OSRM)](http://project-osrm.org/) and added to the dataset.

+++

## Strong IO capabilities

+++

Our dataset is formatted as a 611 MB [Apache Parquet](https://parquet.apache.org/) file, provided here: [https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet](https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet). Alongside JSON, and raw buffers, Awkward can also read Parquet files and Arrow tables. 

Given that this file is so large, let's first look at the _metadata_ with `ak.metadata_from_parquet` to see what we're working with:

```{code-cell} ipython3
metadata = ak.metadata_from_parquet("https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet")
```

Of particular interest here is the `num_row_groups` value. Parquet has the concept of _row groups_: contiguous rows of data in the file, and the smallest granularity that can be read. 

+++

We can also look at the `type` of the data to see the structure of the dataset:

```{code-cell} ipython3
metadata.form.type.show()
```

There are a lot of different columns here (`trip.sec`, `trip.begin.lon`, `trip.payment.fare`, etc.). For this example, we only want a small subset of them. Additionally, we don't need to load _all_ of the data, as we are only interested in a representative sample. Let's use `ak.from_parquet` with the `row_groups` argument to read (download) only a single group, and the `columns` argument to read only the necessary columns:

```{code-cell} ipython3
taxi = ak.from_parquet(
    "https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet",
    row_groups=[0],
    columns=["trip.km", "trip.begin.l*", "trip.end.l*", "trip.path.*"],
)
```

We can look at the `type` of the array to see its structure

```{code-cell} ipython3
taxi.type.show()
```

According to the above, this is an array of `353` elements, and each element is a variable length list (`var`) of records. Each list represents one taxi and each record in each list is a taxi trip.

The `trip` field contains a record with

   * `km`: distance traveled in kilometers
   * `begin.lon`, `begin.lat`: beginning longitude and latitude
   * `end.lon`, `end.lat`: end longitude and latitude
   * `path.londiff`, `path.latdiff`: reconstructed path relative to `begin.lon`, `begin.lat`

+++ {"tags": []}

## Convenient broadcasting

+++

To extract a particular field (column) from an Awkward Array, we can use attribute lookups:

```{code-cell} ipython3
taxi.trip.begin.lat
```

```{code-cell} ipython3
taxi.trip.path.latdiff
```

The `latdiff` and `londiff` columns describe the relative coordinates of waypoints along the taxi routes. In order to plot these routes, we need to add the taxi `lat` and `lon` respectively to these columns. We want this operation to broadcast: each trip has one starting point, but multiple waypoints. 


In NumPy, broadcasting aligns _to the right_, which means that arrays with differing dimensions are made compatible against one another by adding length-1 dimensions to the front of the shape:

```{code-cell} ipython3
x = np.array([1, 2, 3])
y = np.array([
    [4, 5, 6],
    [7, 8, 9]
])
np.broadcast_arrays(x, y)
```

In Awkward, broadcasting aligns _to the left_ by default, which means that length-1 dimensions are added _to the end_ of the shape:

```{code-cell} ipython3
x = ak.Array([1, 2]) # note the missing 3!
y = ak.Array([
    [4, 5, 6],
    [7, 8, 9]
])
ak.broadcast_arrays(x, y)
```

In this instance, we also want broadcasting to align to the left: we want a single starting point to broadcast against multiple waypoints. We can simply add our two arrays together, and Awkward will broadast them correctly:

```{code-cell} ipython3
taxi_trip_lat = taxi.trip.begin.lat + taxi.trip.path.latdiff
taxi_trip_lon = taxi.trip.begin.lon + taxi.trip.path.londiff
```

## Lightweight records

+++

In Awkward, records are lightweight because they can be composed of existing arrays. We can see this with `ak.unzip`, which decomposes an array into its fields:

```{code-cell} ipython3
ak.unzip(taxi.trip)
```

Having computed `taxi_trip_lat` and `taxi_trip_lon`, we might wish to add these as columns to our `taxi` dataset, so that later manipulations are also applied to these values. Here, we can use the subscript operator `[]` to add new fields to our dataset:

```{code-cell} ipython3
taxi[("trip", "path", "lat")] = taxi_trip_lat
taxi[("trip", "path", "lon")] = taxi_trip_lon

taxi.type.show()
```

## Advanced indexing

Let's imagine that we want to plot the three longest routes taken by any taxi in the city. The distance travelled by any taxi is given by `taxi.trip.km`:

```{code-cell} ipython3
taxi.trip.km
```

This array has two dimensions: `353`, indicating the number of taxis, and `var` indicating the number of trips for each taxi. Because we want to find the longest trips amongst all taxis, we can flatten one of the dimensions to produce a list of trips:

```{code-cell} ipython3
trip = ak.flatten(taxi.trip, axis=1)
```

From this list of 1003517 journeys, we can sort by length using `ak.argsort`:

```{code-cell} ipython3
ix_length = ak.argsort(trip.km, ascending=False)
```

Now let's take only the three largest trips

```{code-cell} ipython3
trip_longest = trip[ix_length[:3]]
```

`ipyleaflet` requires a list of coordinate tuples in order to plot a path. Let's concatenate these two arrays together to build a `(16, 2)` array:

```{code-cell} ipython3
lat_lon_taxi_75 = ak.concatenate(
    (trip_longest.path.lat[..., np.newaxis], trip_longest.path.lon[..., np.newaxis]),
    axis=-1
)
```

We can convert this to a list with `tolist()`:

```{code-cell} ipython3
:tags: []

lat_lon_taxi_75.tolist()
```

What does our route look like?

```{code-cell} ipython3
map_taxi_75 = ipl.Map(
    basemap=ipl.basemap_to_tiles(ipl.basemaps.CartoDB.Voyager, "2022-04-08"),
    center=(41.8921, -87.6623),
    zoom=11
)
for route in lat_lon_taxi_75:
    path = ipl.AntPath(
        locations=route.tolist(),
        delay=1000
    )
    map_taxi_75.add_layer(path)
map_taxi_75
```
