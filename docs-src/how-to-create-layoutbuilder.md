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

How to create arrays with LayoutBuilder (more control)
======================================================

What is LayoutBuilder
---------------------

`ak.layout.LayoutBuilder` is the low-level LayoutBuilder that builds layouts, or `ak.layout.Content` arrays. It must be initialized by a JSON string that represents a valid `ak.forms.Form`.

A layout is the composable `ak.layout.Content` elements that determine how an array is structured. The layout may be considered a "low-level" view, as it distinguishes between arrays that have the same logical meaning (i.e. same JSON output and high-level `type`) but different
  * node types, such as `ak.layout.ListArray64` and `ak.layout.ListOffsetArray64`,
  * integer type specialization, such as `ak.layout.ListArray64` and `ak.layout.ListArray32`,
  * or specific values, such as gaps in a `ak.layout.ListArray64`.

`ak.forms.Form` describes a low-level data type or "form". There is an exact one-to-one relationship between each `ak.layout.Content` class and each Form.

`ak.layout.LayoutBuilder` helps you create these "low-level" views that are described by the form. Once the builder is initialized, it can only build a specific view determined by the layout form.

LayoutBuilder vs ArrayBuilder
-----------------------------

The biggest difference between a LayoutBuilder and an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) is that the data types that you can append to the LayoutBuilder are restricted by its Form, while you can append any data types to an ArrayBuilder. The latter flexibility comes with performance limitations.

Appending
---------

  ```{code-cell}
  import awkward as ak
  ```
To create an `ak.layout.LayoutBuilder` a valid `ak.forms.Form` in JSON format is needed to initialize it. This `ak.forms.Form` determines which commands and which data types are accepted by the builder.

Here is an example of a JSON form describing an `ak.layout.UnionArray8_64` array of a [union-type](https://awkward-array.readthedocs.io/en/latest/ak.types.UnionType.html):

```{code-cell}
form = """
{
  "class": "UnionArray8_64",
  "tags": "i8",
  "index": "i64",
  "contents": [
      "float64",
      "bool",
      "int64"
  ],
  "form_key": "node0"
}
  """
```

When a layout builder is created from this form, it cannot be modified. The builder accepts only data types spcified in the form: `float64`, `bool`, or `int64`. The appending data builder methods are restricted to `float64`, `boolean`, and `int64`. The methods have similar to the data type names.

```{code-cell}
builder = ak.layout.LayoutBuilder32(form)
```

A tag is associated with each of the `UnionArray` contents. The tags are contiguous integers, starting with `0` for the first content.

A `tag` command has to be issued prior to each data method:

```{code-cell}
builder.tag(0)
builder.float64(1.1)
builder.tag(1)
builder.boolean(False)
builder.tag(2)
builder.int64(11)
```

 The contents filling order can be arbitrary. `tag` uniquely identifies the content, the next command fills it.

```{code-cell}
builder.tag(0)
builder.float64(2.2)
builder.tag(1)
builder.boolean(False)
builder.tag(0)
builder.float64(2.2)
builder.tag(0)
builder.float64(3.3)
builder.tag(1)
builder.boolean(True)
builder.tag(0)
builder.float64(4.4)
builder.tag(1)
builder.boolean(False)
builder.tag(1)
builder.boolean(True)
builder.tag(0)
builder.float64(-2.2)
```

Snapshot
--------

To turn a `LayoutBuilder` into a layout, call `snapshot`. This is an inexpensive operation (may be done multiple times; the builder is unaffected).

```{code-cell}
layout = builder.snapshot()
layout
```

If you want to use the layout as a high-level array for normal analysis, remember to convert it.

```{code-cell}
array = ak.Array(layout)
array
```

Nested lists
------------

Nested records
--------------

When using a RecordArray form you do not have to specify a field, the fields alternate.

```{code-cell}
form = """
{
"class": "RecordArray",
"contents": {
    "one": "float64",
    "two": "float64"
},
"form_key": "node0"
}
"""
```

```{code-cell}
builder = ak.layout.LayoutBuilder32(form)
```

If record contents have the same type, the fields alternate:

```{code-cell}
builder.float64(1.1)  # "one"
builder.float64(2.2)  # "two"
builder.float64(3.3)  # "one"
builder.float64(4.4)  # "two"
```
