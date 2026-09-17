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

How to convert to/from JSON
===========================

Any JSON data can be converted to Awkward Arrays and any Awkward Arrays can be converted to JSON. Awkward type information, such as the distinction between fixed-size and variable-length lists, is lost in the transformation to JSON, however.

```{code-cell} ipython3
import awkward as ak
import pathlib
```

From JSON to Awkward
--------------------

The function for JSON → Awkward conversion is {func}`ak.from_json`.

It can be given a JSON string:

```{code-cell} ipython3
ak.from_json("[[1.1, 2.2, 3.3], [], [4.4, 5.5]]")
```

or a file name:

```{code-cell} ipython3
!echo "[[1.1, 2.2, 3.3], [], [4.4, 5.5]]" > /tmp/awkward-example-1.json
```

```{code-cell} ipython3
ak.from_json(pathlib.Path("/tmp/awkward-example-1.json"))
```

If the dataset contains a single JSON object, an {class}`ak.Record` is returned, rather than an {class}`ak.Array`.

```{code-cell} ipython3
ak.from_json('{"x": 1, "y": [1, 2], "z": "hello"}')
```

From Awkward to JSON
--------------------

The function for Awkward → JSON conversion is {func}`ak.to_json`.

With one argument, it returns a string.

```{code-cell} ipython3
ak.to_json(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
```

But if a `destination` is given, it is taken to be a filename for output.

```{code-cell} ipython3
ak.to_json(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]), "/tmp/awkward-example-2.json")
```

```{code-cell} ipython3
!cat /tmp/awkward-example-2.json
```

Conversion of different types
-----------------------------

All of the rules that apply for Python objects in {func}`ak.from_iter` and {func}`ak.to_list` apply to {func}`ak.from_json` and {func}`ak.to_json`, replacing builtin Python types for JSON types. (One exception: JSON has no equivalent of a Python tuple.)

+++

Performance
-----------

Since Awkward Array internally uses [RapidJSON](https://rapidjson.org/) to simultaneously parse and convert the JSON string, {func}`ak.from_json` and {func}`ak.to_json` should always be faster and use less memory than {func}`ak.from_iter` and {func}`ak.to_list`. Don't convert JSON strings into or out of Python objects for the sake of converting them as Python objects: use the JSON converters directly.
