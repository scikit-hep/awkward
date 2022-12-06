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

# 10 minutes to Awkward Array

This is a short, tutorial-style introduction to Awkward Array, aimed towards new users. For details of how to perform specific operations in Awkward Array, e.g. _filtering data_, see the user-guide [index](index.md), or use the search tool to identify relevant pages.

The City of Chicago has a [Data Portal](https://data.cityofchicago.org/) with lots of interesting datasets. This guide uses a dataset of [Chicago taxi trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) taken from 2019 through 2021 (3 years).

The dataset that the Data Portal provides has trip start and stop points as longitude, latitude pairs, as well as start and end times (date-stamps), payment details, and the name of each taxi company. To make the example more interesting, an estimated route of each taxi trip has been computed by [Open Source Routing Machine (OSRM)](http://project-osrm.org/) and added to the dataset.

In this guide, we'll look at how to manipulate a jagged dataset to plot taxi routes in Chicago.


## Loading the dataset

Our dataset is formatted as a 611 MB [Apache Parquet](https://parquet.apache.org/) file, provided [here](https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet). Alongside JSON, and raw buffers, Awkward can also read Parquet files and Arrow tables.

Given that this file is so large, let's first look at the *metadata* with `ak.metadata_from_parquet` to see what we're working with:

```{code-cell} ipython3
:tags: [hide-cell]

%config InteractiveShell.ast_node_interactivity = "last_expr_or_assign"
```

```{code-cell} ipython3
import numpy as np
import awkward as ak

metadata = ak.metadata_from_parquet(
    "https://pivarski-princeton.s3.amazonaws.com/chicago-taxi.parquet"
)
```

Of particular interest here is the `num_row_groups` value. Parquet has the concept of *row groups*: contiguous rows of data in the file, and the smallest granularity that can be read.

We can also look at the `type` of the data to see the structure of the dataset:

```{code-cell} ipython3
metadata["form"].type.show()
```

There are a lot of different columns here (`trip.sec`, `trip.begin.lon`, `trip.payment.fare`, etc.). For this example, we only want a small subset of them. Additionally, we don't need to load *all* of the data, as we are only interested in a representative sample. Let's use `ak.from_parquet` with the `row_groups` argument to read (download) only a single group, and the `columns` argument to read only the necessary columns.

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

-   `km`: distance traveled in kilometers
-   `begin`: record containing longitude and latitude of the trip start point
-   `end`: record containing longitude and latitude of the trip end point
-   `path`: list of records containing the relative longitude and latitude of the trip waypoints

+++

## Reconstructing the routes
In order to plot the taxi routes, we can use the waypoints given by the `path` field. However, these waypoints are relative to the trip start point; we need to add the starting position to these relative positions in order to plot them on a map.

+++

The fields of a record can be accessed with attribute notation:

```{code-cell} ipython3
taxi.trip
```

or using subscript notation:

```{code-cell} ipython3
taxi["trip"]
```

Field lookup can be nested, e.g. with attribute notation:

```{code-cell} ipython3
taxi.trip.path
```

or with subscript notation:

```{code-cell} ipython3
taxi["trip", "path"]
```

Let's look at two fields of interest, `path.latdiff`, and `begin.lat`, and their types:

```{code-cell} ipython3
taxi.trip.path.latdiff
```

```{code-cell} ipython3
taxi.trip.path.latdiff.type.show()
```

```{code-cell} ipython3
taxi.trip.begin.lat
```

```{code-cell} ipython3
taxi.trip.begin.lat.type.show()
```

Clearly, these two arrays have _different_ dimensions. When we add them together, we want the operation to broadcast: each trip has one starting point, but multiple waypoints. In NumPy, broadcasting aligns *to the right*, which means that arrays with differing dimensions are made compatible against one another by adding length-1 dimensions to the front of the shape:

```{code-cell} ipython3
x = np.array([1, 2, 3])
y = np.array(
    [
        [4, 5, 6],
        [7, 8, 9],
    ]
)
np.broadcast_arrays(x, y)
```

In Awkward, broadcasting aligns *to the left* by default, which means that length-1 dimensions are added *to the end* of the shape:

```{code-cell} ipython3
x = ak.Array([1, 2])  # note the missing 3!
y = ak.Array(
    [
        [4, 5, 6],
        [7, 8, 9],
    ]
)
ak.broadcast_arrays(x, y)
```

In this instance, we also want broadcasting to align to the left: we want a single starting point to broadcast against multiple waypoints. We can simply add our two arrays together, and Awkward will broadast them correctly.

```{code-cell} ipython3
taxi_trip_lat = taxi.trip.begin.lat + taxi.trip.path.latdiff
taxi_trip_lon = taxi.trip.begin.lon + taxi.trip.path.londiff
```

## Storing the routes

Having computed `taxi_trip_lat` and `taxi_trip_lon`, we might wish to add these as fields to our `taxi` dataset, so that later manipulations are also applied to these values. Here, we can use the subscript operator `[]` to add new fields to our dataset.

```{code-cell} ipython3
taxi[("trip", "path", "lat")] = taxi_trip_lat
taxi[("trip", "path", "lon")] = taxi_trip_lon
```

Note that fields cannot be set using attribute notation.

+++

We can see the result of adding these fields:

```{code-cell} ipython3
taxi.type.show()
```

In Awkward, records are lightweight because they can be composed of existing arrays. We can see this with `ak.fields`, which returns a list containing the field names of a record:

```{code-cell} ipython3
ak.fields(taxi.trip.path)
```

and `ak.unzip`, which decomposes an array into the corresponding field values:

```{code-cell} ipython3
ak.unzip(taxi.trip.path)
```

## Finding the longest routes

Let's imagine that we want to plot the three longest routes taken by any taxi in the city. The distance travelled by any taxi is given by `taxi.trip.km`:

```{code-cell} ipython3
taxi.trip.km
```

This array has two dimensions: `353`, indicating the number of taxis, and `var` indicating the number of trips for each taxi. Because we want to find the longest trips amongst all taxis, we can flatten one of the dimensions using `ak.flatten` to produce a list of trips.

```{code-cell} ipython3
trip = ak.flatten(taxi.trip, axis=1)
```

From this list of 1003517 journeys, we can sort by length using `ak.argsort`.

```{code-cell} ipython3
ix_length = ak.argsort(trip.km, ascending=False)
```

Now let's take only the three largest trips.

```{code-cell} ipython3
trip_longest = trip[ix_length[:3]]
```

## Plotting the longest routes

`ipyleaflet` requires a list of coordinate tuples in order to plot a path. Let's stack these two arrays together to build a `(16, 2)` array.

```{code-cell} ipython3
lat_lon_taxi_75 = ak.concatenate(
    (trip_longest.path.lat[..., np.newaxis], trip_longest.path.lon[..., np.newaxis]),
    axis=-1,
)
```

We can convert this to a list with `to_list()`:

```{code-cell} ipython3
lat_lon_taxi_75.to_list()
```

What does our route look like?

```{code-cell} ipython3
import ipyleaflet as ipl

map_taxi_75 = ipl.Map(
    basemap=ipl.basemap_to_tiles(ipl.basemaps.CartoDB.Voyager, "2022-04-08"),
    center=(41.8921, -87.6623),
    zoom=11,
)
for route in lat_lon_taxi_75:
    path = ipl.AntPath(locations=route.to_list(), delay=1000)
    map_taxi_75.add_layer(path)
map_taxi_75
```
