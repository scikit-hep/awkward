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

How to examine a single item in detail
======================================

It's often useful to pull out a single item from an array to inspect its contents, particularly in the early stages of a data analysis, to get a sense of the data's structure. This tutorial shows how to extract one item from an Awkward Array and examine it in different ways.

For this example, we'll to use the Chicago taxi trips dataset from [10 minutes to Awkward Array](https://awkward-array.org/doc/main/getting-started/10-minutes-to-awkward-array.html). Recall that this dataset includes information about trips by various taxis collected over a few years, enriched with GPS path data.

+++

## Loading the dataset

First, let's load the dataset using the {func}`ak.from_parquet` function. We will only load the first row group, for the sake of this demonstration:

```{code-cell} ipython3
import awkward as ak

url = "https://zenodo.org/records/14537442/files/chicago-taxi.parquet"
taxi = ak.from_parquet(
    url,
    row_groups=[0],
    columns=["trip.km", "trip.begin.l*", "trip.end.l*", "trip.path.*"],
)
```

## What is a single item?

The first "item" of this dataset could be a single taxi, which comprises many trips.

```{code-cell} ipython3
single_taxi = taxi[5]
single_taxi
```

Or it could be a single trip.

```{code-cell} ipython3
single_trip = single_taxi.trip[5]
single_trip
```

Or it could be a single latitude, longitude position along the path.

```{code-cell} ipython3
single_trip.path
```

```{code-cell} ipython3
single_point = single_trip.path[5]
single_point
```

```{code-cell} ipython3
print(f"longitude: {single_trip.begin.lon + single_point.londiff:.3f}")
print(f"latitude:  {single_trip.begin.lat + single_point.latdiff:.3f}")
```

In Jupyter notebooks (and this documentation), the array contents are presented in a multi-line format with the data type below a dashed line.

## Standard Python `repr`

In a Python prompt, the format is more concise:

```{code-cell} ipython3
print(f"{single_taxi!r}")
```

```{code-cell} ipython3
print(f"{single_trip!r}")
```

```{code-cell} ipython3
print(f"{single_point!r}")
```

The long form can be obtained in a Python prompt with the `show` method:

```{code-cell} ipython3
single_taxi.show()
```

```{code-cell} ipython3
single_trip.show()
```

```{code-cell} ipython3
single_point.show()
```

## The `show` method

The `show` method can take a `type=True` argument to include the type as well (at the top this time, because values are presented in the "most valuable real estate," which is the bottom of a print-out in the terminal, but the top in a Jupyter notebook).

```{code-cell} ipython3
single_point.show(type=True)
```

Types also have a `show` method, so if you _only_ want the type, you can do

```{code-cell} ipython3
single_trip.type.show()
```

If you need to get this as a string or pass it to an output other than `sys.stdout`, use the `stream` parameter.

```{code-cell} ipython3
single_point.show(stream=None)
```

## Using `to_list` and Python’s `pprint` for a detailed view

The `repr` and `show` representations print into a restricted space: 1 line (80 characters) for `repr`, and 20 lines (80 character width) for `show` without `type=True`. To do this, they replace data with ellipses (`...`) until it fits.

You might want to ensure that you see everything. One way to do that is to turn the data into Python objects with {func}`ak.to_list` (or `to_list` or `tolist` as a method) and pretty-print them with Python's `pprint`.

```{code-cell} ipython3
import pprint

trip_list = ak.to_list(single_trip)
pprint.pprint(trip_list)
```

Keep in mind that if you don't slice a small enough section of data, your terminal or Jupyter notebook may be overwhelmed with output!

## Viewing data as JSON

Another way you can dump everything is to convert the data to JSON with {func}`ak.to_json`.

```{code-cell} ipython3
print(ak.to_json(single_trip))
```

That's not very readable, so we'll pass `num_indent_spaces=4` to add newlines and indentation, and `num_readability_spaces=1` to add spaces after commas (`,`) and colons (`:`).

```{code-cell} ipython3
print(ak.to_json(single_trip, num_indent_spaces=4, num_readability_spaces=1))
```

{func}`ak.to_json` is also one of the bulk output methods, so it can write data to a file, as a single JSON object or as `line_delimited` JSON.
