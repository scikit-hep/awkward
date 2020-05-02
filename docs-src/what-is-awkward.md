---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.4.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

What is an Awkward Array?
=========================

Arrays are fundamental to computing, and libraries like [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) make it easy to interact with arrays in Python. However, these arrays are rectilinear—tables in two dimensions or tensors in more—while real data are nested tree-like structures, often represented by [JSON](https://www.json.org/).

Sometimes, data start off as JSON for generality, but eventually need to be flattened into a rectilinear shape for large-scale processing. In high-energy particle physics, the datasets are always large enough and complex enough that neither JSON nor a separate "cleaning" step are feasible.

Awkward Array was developed for datasets that are, well, awkward to deal with any other way.

Consider, for example, this GeoJSON of Chicago bike paths:

```{code-cell}
import urllib.request
import json

url = "https://raw.githubusercontent.com/Chicago/osd-bike-routes/master/data/Bikeroutes.geojson"
bikeroutes_json = urllib.request.urlopen(url).read()
bikeroutes_pyobj = json.loads(bikeroutes_json)
```

It's a complicated dataset with street names and variable-length polylines. But as an Awkward Array,

```{code-cell}
import awkward1 as ak
bikeroutes = ak.Record(bikeroutes_pyobj)
bikeroutes
```

we can slice it as we might a NumPy array.

```{code-cell}
bikeroutes["features", "properties", "STREET"]
```

In the above, we see that there are 1061 streets and the first is named `'W FULLERTON AVE`. The longitude and latitude coordinates can be obtained with a slice on coordinates.

```{code-cell}
longitude = bikeroutes["features", "geometry", "coordinates", ..., 0]
latitude = bikeroutes["features", "geometry", "coordinates", ..., 1]
longitude, latitude
```

Even though `longitude` and `latitude` are arrays of lists of lists, we can do NumPy-like math that preserves their structures.

```{code-cell}
km_east = (longitude - ak.mean(longitude)) * 82.7
km_north = (latitude - ak.mean(latitude)) * 111.1
km_east, km_north
```

We can even drop the first (`1:`) and last (`:-1`) of each segment in those variable-length polylines to compute differences, and from that, distances. The length of each street is a sum over a variable number of segments.

```{code-cell}
import numpy as np
segment_length = np.sqrt((km_east[:, :, 1:] - km_east[:, :, :-1])**2 +
                         (km_north[:, :, 1:] - km_north[:, :, :-1])**2)
street_length = ak.sum(segment_length, axis=-1)
street_length
```

And thus,

```{code-cell}
for i in range(10):
    print(bikeroutes.features.properties.STREET[i], "\t\t", street_length[i, 0])
```


