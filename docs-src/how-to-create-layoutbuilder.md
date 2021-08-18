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

A layout is the composable #ak.layout.Content elements that determine how an #ak::Array is structured. The layout may be considered a "low-level" view, as it distinguishes between arrays that have the same logical meaning (i.e. same JSON output and high-level #type) but different
  * node types, such as #ak.layout.ListArray64 and #ak.layout.ListOffsetArray64,
  * integer type specialization, such as #ak.layout.ListArray64 and #ak.layout.ListArray32,
  * or specific values, such as gaps in a #ak.layout.ListArray64.

ak::LayoutBuilder can help you to create these "low-level" views that are described by a ak::Form. Once the builder is initialized, it can only build a specific view determined by the layout form.

The biggest difference between a #ak::LayoutBuilder and an [ak.ArrayBuilder](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ArrayBuilder.html) is that the data types that you can append to the #ak::LayoutBuilder are restricted by its #ak::Form, while you can append any data types to an #ak::ArrayBuilder. The latter flexibility comes with performance limitations.

+++

Appending
---------

  ```{code-cell}
  import awkward as ak
  ```
To create an #ak::LayoutBuilder an #ak::Form in JSON format is needed to initialize it. This #ak::Form determines which commands and which data types are accepted by the builder.

Here is an example of a form describing an #ak.layout.UnionArray8_64 of a [union-type](https://awkward-array.readthedocs.io/en/latest/ak.types.UnionType.html):

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

When a builder is created, it will accept only "float64", "bool", and "int64" data. The builder data methods to append these types have similar to the types names: `float64`, `boolean`, and `int64`.

```{code-cell}
builder = ak.layout.LayoutBuilder(form)
```

Each of the union array contents has a tag associated with it. The tags are continues integers, the first content tag is `0`. You have to issue a `tag` command prior to each data method. That uniquely identifies the content to which the data is appended:

```{code-cell}
builder.tag(0)
builder.float64(1.1)
builder.tag(1)
builder.boolean(False)
builder.tag(2)
builder.int64(11)
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

To turn a #ak::LauoutBuilder into a layout, call `snapshot`. This is an inexpensive operation (may be done multiple times; the builder is unaffacted).

```{code-cell}
layout = builder.snapshot()
layout
```

If you want to use the layout as a high-level array for normal analysis, remember to convert it.

```{code-cell}
array = ak.Array(layout)
array
```
