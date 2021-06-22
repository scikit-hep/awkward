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

How to create arrays of records
===============================

In Awkward Array, a "record" is a structure containing a fixed-length set of typed, possibly named fields. This is a "struct" in C or an "object" in Python (though the association of executable methods to record types is looser than the binding of methods to classes in object oriented languages).

All methods in Awkward Array are implemented as "[structs of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA)," rather than arrays of structs, so making and breaking records are inexpensive operations that you can perform frequently in data analysis.

```{code-cell}
import awkward as ak
import numpy as np
```

From a list of Python dicts
---------------------------

Records have a natural representation in JSON and Python as dicts, but only if all dicts in a series have the same set of field names. The [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) invokes [ak.from_iter](https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_iter.html) whenever presented with a list (or other non-string, non-dict iterable).

```{code-cell}
python_dicts = [
    {"x": 1, "y": 1.1, "z": "one"},
    {"x": 2, "y": 2.2, "z": "two"},
    {"x": 3, "y": 3.3, "z": "three"},
    {"x": 4, "y": 4.4, "z": "four"},
    {"x": 5, "y": 5.5, "z": "five"},
]
python_dicts
```

```{code-cell}
awkward_array = ak.Array(python_dicts)
awkward_array
```

It is important that all of the dicts in the series have the same set of field names, since Awkward Array has to identify all of the records as having a single type:

```{code-cell}
awkward_array.type
```

That is to say, an array of records is not a mapping from one type to another, such as from strings to numbers. The above record has exactly three fields: _x_, _y_, and _z_, and they have fixed types: `int64`, `float64`, and `string`.

   * Python dicts are sometimes used as records, sometimes used as mappings.
   * JSON objects are most often used as records, but sometimes mappings (with a restricted key type: JSON object keys must be strings).
   * Records in an Awkward Array must always be records.

```{code-cell}

```
