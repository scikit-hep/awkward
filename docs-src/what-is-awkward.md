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

What is an Awkward Array?
=========================

Efficiency and generality
-------------------------

Arrays are the most efficient data structures for sequential numeric processing, and [NumPy](https://numpy.org/) makes it easy to interact with arrays in Python. However, NumPy's arrays are rectangular tables or tensors that cannot express variable-length structures.

General tree-like data are often expressed using [JSON](https://www.json.org/), but at the expense of memory use and processing speed.

Awkward Arrays are general tree-like data structures, like JSON, but contiguous in memory and operated upon with compiled, vectorized code like NumPy. They're basic building blocks for data analyses that are, well, more awkward than those involving neat tables.

This library was originally developed for high-energy particle physics. Particle physics datasets have rich data structures that usually can't be flattened into rectangular arrays, but physicists need to process them efficiently because the datasets are enormous. Awkward Arrays combine generic data structures with high-performance number-crunching.

Let's illustrate this with a non-physics dataset: maps of bike routes in my hometown of Chicago. You can also follow this [as a video tutorial](https://youtu.be/WlnUF3LRBj4?t=431).

<iframe width="560" height="315" src="https://www.youtube.com/embed/WlnUF3LRBj4?start=431" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

JSON to array
-------------

Here is a [GeoJSON of bike paths](https://github.com/Chicago/osd-bike-routes/blob/master/data/Bikeroutes.geojson) of bike paths throughout the city of Chicago.

<div style="margin-right: 15px">
  <iframe class="render-viewer " src="https://render.githubusercontent.com/view/geojson?commit=5f556dcd4ed54f5f5c926c01c34ebc6261ec7d34&amp;enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4368696361676f2f6f73642d62696b652d726f757465732f356635353664636434656435346635663563393236633031633334656263363236316563376433342f646174612f42696b65726f757465732e67656f6a736f6e&amp;nwo=Chicago%2Fosd-bike-routes&amp;path=data%2FBikeroutes.geojson&amp;repository_id=8065965&amp;repository_type=Repository#292b99b1-4187-4d13-8b0d-ce8fa6eb4ab1" sandbox="allow-scripts allow-same-origin allow-top-navigation" title="File display" width="100%" height="400px"></iframe>
</div>

If you dig into the JSON, you'll see that it contains street names, metadata, and longitude, latitude coordinates all along the bike paths.

Start by loading them into Python as Python objects,

```{code-cell}
import urllib.request
import json

url = "https://raw.githubusercontent.com/Chicago/osd-bike-routes/master/data/Bikeroutes.geojson"
bikeroutes_json = urllib.request.urlopen(url).read()
bikeroutes_pyobj = json.loads(bikeroutes_json)
```

and then as an Awkward Array (actually an [ak.Record](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Record.html) because the top-level construct is a JSON object).

```{code-cell}
import awkward as ak

bikeroutes = ak.from_json(bikeroutes_json)
# Alternatively, bikeroutes = ak.Record(bikeroutes_pyobj)
bikeroutes
```

We only see a part of the data and its type if we don't deliberately expand it out.

Data types
----------

To get a full view of the type (Awkward's equivalent of a NumPy dtype + shape), use the [ak.type](https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html) function. The display format adheres to [Datashape syntax](https://datashape.readthedocs.io/), when possible.

```{code-cell}
ak.type(bikeroutes)
```

In the above, `{"field name": type, ...}` denotes a record structure, which can be nested, and `var` indicates variable-length lists. The `"coordinates"` (at the end) are `var * var * var * float64`, lists of lists of lists of numbers, and any of these lists can have an arbitrary length.

In addition, there are strings (variable-length lists interpreted as text) and "option" types, meaning that values are allowed to be null.

Slicing
-------

[NumPy-like slicing](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html#ak-array-getitem) extracts structures within the array. The slice may consist of integers, ranges, and many other slice types, like NumPy, and commas indicate different slices applied to different dimensions. Since the bike routes dataset contains records, we can use strings to select nested fields sequentially.

```{code-cell}
bikeroutes["features", "geometry", "coordinates"]
```

Alternatively, we could use dots for record field specifiers (if the field names are syntactically allowed in Python):

```{code-cell}
bikeroutes.features.geometry.coordinates
```

Slicing by field names (even if the records those fields belong to are nested within lists) slices across all elements of the lists. We can pick out just one object by putting integers in the square brackets:

```{code-cell}
bikeroutes["features", "geometry", "coordinates", 100, 0]
```

or

```{code-cell}
bikeroutes.features.geometry.coordinates[100, 0]
```

or even

```{code-cell}
bikeroutes.features[100].geometry.coordinates[0]
```

(The strings that select record fields may be placed before or after integers and other slice types.)

To get full detail of one structured object, we can use the [ak.to_list](https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_list.html) function, which converts Awkward records and lists into Python dicts and lists.

```{code-cell}
ak.to_list(bikeroutes.features[751])
```

Looking at one record in full detail can make it clear why, for instance, the "coordinates" field contains lists of lists of lists: they are path segments that collectively form a route, and there are many routes, each associated with a named street. This item, number `751`, is Martin Luther King Drive, a route described by 7 segments. (Presumably, you have to pick up your bike and walk it.)

Variable-length lists
---------------------

The last dimension of these lists always happens to have length 2. This is because it represents the longitude and latitude of each point along a path. You can see this with the [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html) function:

```{code-cell}
ak.num(bikeroutes.features[751].geometry.coordinates, axis=2)
```

The `axis` is the depth at which this function is applied; the above could alternatively have been `axis=-1` (deepest), and [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html) at less-deep `axis` values tells us the number of points in each segment:

```{code-cell}
ak.num(bikeroutes.features[751].geometry.coordinates, axis=1)
```

and the number of points:

```{code-cell}
ak.num(bikeroutes.features[751].geometry.coordinates, axis=0)
```

By verifying that all lists at this depth have length 2,

```{code-cell}
ak.all(ak.num(bikeroutes.features.geometry.coordinates, axis=-1) == 2)
```

we can be confident that we can select item `0` and item `1` without errors. Note that this is a major difference between variable-length lists and rectilinear arrays: in NumPy, a given index either exists for all nested lists or for none of them. For variable-length lists, we have to check (or ensure it with another selection).

Array math
----------

We now know that the `"coordinates"` are longitude-latitude pairs, so let's pull them out and name them as such. Item `0` of each of the deepest lists is the longitude and item `1` of each of the deepest lists is the latitude. We want to leave the structure of all lists other than the deepest untouched, which would mean a complete slice (colon `:` by itself) at each dimension except the last, but we can also use the ellipsis (`...`) shortcut from NumPy.

```{code-cell}
longitude = bikeroutes.features.geometry.coordinates[..., 0]
latitude = bikeroutes.features.geometry.coordinates[..., 1]
longitude, latitude
```

Note that if we wanted to do this with Python objects, the above would have required many "append" operations in nested "for" loops. As Awkward Arrays, it's just a slice.

Now that we have arrays of pure numbers (albeit inside of variable-length nested lists), we can run NumPy functions on them. For example,

```{code-cell}
import numpy as np

np.add(longitude, 180)
```

rotates the longitude points 180 degrees around the world while maintaining the triply nested structure. Any "[universal function](https://numpy.org/doc/stable/reference/ufuncs.html)" (ufunc) will work, including ufuncs from libraries other than NumPy (such as SciPy, or a domain-specific package). Simple NumPy functions like addition have the usual shortcuts:

```{code-cell}
longitude + 180
```

In addition, some functions other than ufuncs have an Awkward equivalent, such as [ak.mean](https://awkward-array.readthedocs.io/en/latest/_auto/ak.mean.html), which is the equivalent of NumPy's [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) (not a ufunc because it takes a whole array and returns one value).


```{code-cell}
ak.mean(longitude)
```

Using an extension mechanism within NumPy ([introduced in NumPy 1.17](https://numpy.org/devdocs/release/1.17.0-notes.html#numpy-functions-now-always-support-overrides-with-array-function)), we can use [ak.mean](https://awkward-array.readthedocs.io/en/latest/_auto/ak.mean.html) and [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) interchangeably.

```{code-cell}
np.mean(longitude)
```

Awkward functions have all or most of the same arguments as their NumPy equivalents. For instance, we can compute the mean along an axis, such as `axis=1`, which gives us the mean longitude of each path, rather than a single mean of all points.

```{code-cell}
np.mean(longitude, axis=1)
```

To focus our discussion, let's say that we're trying to find the length of each path in the dataset. To do this, we need to convert the degrees longitude and latitude into common distance units, and to work with smaller numbers, we'll start by subtracting the mean.

At Chicago's latitude, one degree of longitude is 82.7 km and one degree of latitude is 111.1 km, which we can use as conversion factors.

```{code-cell}
km_east = (longitude - np.mean(longitude)) * 82.7 # km/deg
km_north = (latitude - np.mean(latitude)) * 111.1 # km/deg
km_east, km_north
```

To find distances between points, we first have to pair up points with their neighbors. Each path segment of $N$ points has $N-1$ pairs of neighbors. We can construct these pairs by making two partial copies of each list, one with everything except the first element and the other with everything except the last element, so that original index $i$ can be compared with original index $i+1$.

In plain NumPy, you would express it like this:

```{code-cell}
path = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
path[1:] - path[:-1]
```

The `array[1:]` has the first element dropped and the `array[:-1]` has the last element dropped, so their differences are the 8 distances between each of the 9 points in the original `array`. In this example, all differences are `1.1`.

Here's what that looks like for the first segment of the first bike path in our sample:

```{code-cell}
km_east[0, 0, 1:], km_east[0, 0, :-1]
```

and their differences are:

```{code-cell}
km_east[0, 0, 1:] - km_east[0, 0, :-1]
```

If we can do it for one list, we can do it for all of them by swapping index `0` with slice `:` in the first two dimensions.

```{code-cell}
km_east[:, :, 1:] - km_east[:, :, :-1]
```

This expression subtracts pairs of neighboring points in all lists, each with a different length, maintaining the segments-within-paths structure.

Now that we know how to compute differences in $x$ (`km_east`) and $y$ (`km_north`) individually, we can compute distances using the distance formula: $\sqrt{(x_i - x_{i + 1})^2 + (y_i - y_{i + 1})^2}$.

```{code-cell}
segment_length = np.sqrt(
    ( km_east[:, :, 1:] -  km_east[:, :, :-1])**2 +
    (km_north[:, :, 1:] - km_north[:, :, :-1])**2
)
segment_length
```

Going back to our example of Martin Luther King Drive, these pairwise distances are

```{code-cell}
ak.to_list(segment_length[751])
```

for each of the segments in this discontiguous path. Some of these segments had only two longitude, latitude points, and hence they have only one distance (single-element lists).

To make path distances from the pairwise distances, we need to add them up. There's an [ak.sum](https://awkward-array.readthedocs.io/en/latest/_auto/ak.sum.html) (equivalent to [np.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)) that we can use with `axis=-1` to add up the innermost lists.

For Martin Luther King Drive, this is

```{code-cell}
ak.to_list(ak.sum(segment_length[751], axis=-1))
```

and in general, it's

```{code-cell}
path_length = np.sum(segment_length, axis=-1)
path_length
```

Notice that `segment_length` has type

```{code-cell}
ak.type(segment_length)
```

and `path_length` has type

```{code-cell}
ak.type(path_length)
```

The `path_length` has one fewer `var` dimension because we have summed over it. We can further sum over the discontiguous curves that 11 of the streets have to get total lengths.

Since there are multiple paths for each bike route, we sum up the innermost dimension again:


```{code-cell}
route_length = np.sum(path_length, axis=-1)
route_length
```

Now there's exactly one of these for each of the 1061 streets.

```{code-cell}
for i in range(10):
    print(bikeroutes.features.properties.STREET[i], "\t\t", route_length[i])
```

This would have been incredibly awkward to write using only NumPy, and slow if executed in Python loops.

Performance
-----------

The full analysis, expressed in **Python for loops**, would be:

```{code-cell}
%%timeit

route_length = []
for route in bikeroutes_pyobj["features"]:
    path_length = []
    for segment in route["geometry"]["coordinates"]:
        segment_length = []
        last = None
        for lng, lat in segment:
            km_east = lng * 82.7
            km_north = lat * 111.1
            if last is not None:
                dx2 = (km_east - last[0])**2
                dy2 = (km_north - last[1])**2
                segment_length.append(np.sqrt(dx2 + dy2))
            last = (km_east, km_north)
        path_length.append(sum(segment_length))
    route_length.append(sum(route_length))
```

whereas for **Awkward Arrays**, it is:

```{code-cell}
%%timeit

km_east = bikeroutes.features.geometry.coordinates[..., 0] * 82.7
km_north = bikeroutes.features.geometry.coordinates[..., 1] * 111.1

segment_length = np.sqrt((km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
                         (km_north[:, :, 1:] - km_north[:, :, :-1])**2)

path_length = np.sum(segment_length, axis=-1)
route_length = np.sum(path_length, axis=-1)
```

In addition to being more concise, the latter is typically 5‒8× faster, especially when we scale to ever-larger problems:

![](img/bikeroutes-scaling.svg)

The reasons for this speedup are all related to Awkward Array's data structure, that it is more suited to structured numerical math than Python objects. Like a NumPy array, its numerical data are packed in memory-contiguous arrays of homogeneous type, which means that

   * only a single block of memory needs to be fetched from memory into the CPU cache (no "pointer chasing"),
   * data for fields other than the one being operated upon are not in the same buffer, so they don't even need to be loaded ("columnar," rather than "record-oriented"),
   * the data type can be evaluated once before applying a precompiled opeation to a whole array buffer, rather than once before each element of a Python list.

This memory layout is especially good for applying one operation on all values in the array, thinking about the result, and then applying another. This is the "interactive" style of data analysis that you're probably familiar with from NumPy and Pandas, especially if you use Jupyter notebooks. It does have a performance cost, however: array buffers need to be allocated and filled after each step of the process, and some of those might never be used again.

Just as NumPy can be accelerated by just-in-time compiling your code with [Numba](http://numba.pydata.org/), Awkward Arrays can be accelerated in the same way. The speedups described on Numba's website are possible because they avoid creating temporary, intermediate arrays and flushing the CPU cache with multiple passes over the same data. The Numba-accelerated equivalent of our bike routes example looks very similar to the pure Python code:

```{code-cell}
import numba as nb

@nb.jit
def compute_lengths(bikeroutes):
    route_length = np.zeros(len(bikeroutes.features))
    for i in range(len(bikeroutes.features)):
        for path in bikeroutes.features[i].geometry.coordinates:
            first = True
            last_east, last_north = 0.0, 0.0
            for lng_lat in path:
                km_east = lng_lat[0] * 82.7
                km_north = lng_lat[1] * 111.1
                if not first:
                    dx2 = (km_east - last_east)**2
                    dy2 = (km_north - last_north)**2
                    route_length[i] += np.sqrt(dx2 + dy2)
                first = False
                last_east, last_north = km_east, km_north
    return route_length

compute_lengths(bikeroutes)
```

But it runs 250× faster than the pure Python code:

```{code-cell}
%%timeit

compute_lengths(bikeroutes)
```

(Note that these are microseconds, not milliseconds.)

This improvement is due to a combination of streamlined data structures, precompiled logic, and minimizing the number of passes over the data. We haven't even taken advantage of multithreading yet, which can multiply this speedup by (up to) the number of CPU cores your computer has. (See Numba's [parallel range](https://numba.pydata.org/numba-doc/0.11/prange.html), [multithreading](https://numba.pydata.org/numba-doc/latest/user/threading-layer.html), and [nogil mode](https://numba.pydata.org/numba-doc/latest/user/jit.html#nogil) for more.)

Internal structure
------------------

It's possible to peek into this columnar structure (or manipulate it, if you're a developer) by accessing the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html)'s layout. All of the columnar buffers are accessible this way.

If you look carefully at the following, you'll see that all values for each field is in a separate buffer; the last of these is the longitude, latitude coordinates.

```{code-cell}
bikeroutes.layout
```

Compatibility
-------------

The Awkward Array library is not intended to replace your data analysis tools. It adds one key feature: the ability to manipulate JSON-like data structures with NumPy-like idioms. It "plays well" with the scientific Python ecosystem, providing functions to convert arrays into forms recognized by other libraries and adheres to standard protocols for sharing data.

They can be converted to and from [Apache Arrow](https://arrow.apache.org/):

```{code-cell}
ak.to_arrow(bikeroutes.features).type
```

To and from [Parquet files](https://parquet.apache.org/) (through pyarrow):

```{code-cell}
ak.to_parquet(bikeroutes.features, "/tmp/bikeroutes.parquet")
```

To and from JSON:

```{code-cell}
ak.to_json(bikeroutes.features)[:100]
```

To Pandas:

```{code-cell}
ak.to_pandas(bikeroutes.features)
```

And to NumPy, if the arrays are first padded to be rectilinear:

```{code-cell}
ak.to_numpy(
    ak.pad_none(
        ak.pad_none(
            bikeroutes.features.geometry.coordinates, 1980, axis=2
        ), 7, axis=1
    )
)
```

Where to go next
----------------

The rest of these tutorials show how to use Awkward Array with various libraries, as well as how to do things that only Awkward Array can do. They are organized by task: see the left-bar (≡ button on mobile) for what you're trying to do. If, however, you're looking for documentation on a specific function, see the Python and C++ references below.

<table style="margin-top: 30px">
  <tr>
    <td width="50%" valign="top" align="center">
      <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/main/docs-img/panel-sphinx.png" width="80%">
      </a>
      <p align="center" style="margin-top: 10px"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/index.html">
        Python<br>API reference
        </a>
      </b></p>
    </td>
    <td width="50%" valign="top" align="center">
      <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        <img src="https://github.com/scikit-hep/awkward-1.0/raw/main/docs-img/panel-doxygen.png" width="80%">
      </a>
      <p align="center" style="margin-top: 10px"><b>
        <a href="https://awkward-array.readthedocs.io/en/latest/_static/index.html">
        C++<br>API reference
        </a>
      </b></p>
    </td>
  </tr>
</table>
