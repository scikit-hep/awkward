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

A layout consists of composable `ak.layout.Content` elements that determine how an array is structured. The layout may be considered a "low-level" view, as it distinguishes between arrays that have the same logical meaning (i.e. same JSON output and high-level `type`) but different
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

To fill data inside of a list use the following commands:

  * `begin_list`/`end_list`

Here is an example of a list offset array form:

```{code-cell}
form = """
{
  "class": "ListOffsetArray64",
  "offsets": "i64",
  "content": "float64",
  "form_key": "node0"
}
"""
```
Create a builder from the form:

```{code-cell}
builder = ak.layout.LayoutBuilder32(form)
```

and append the data between `begin_list` and `end_list`:

```{code-cell}
builder.begin_list()
builder.float64(1.1)
builder.float64(2.2)
builder.float64(3.3)
builder.end_list()
```

To append an empty list:

```{code-cell}
builder.begin_list()
builder.end_list()
```

and continue:

```{code-cell}
builder.begin_list()
builder.float64(4.4)
builder.float64(5.5)
builder.end_list()
```
Remember, you can taka a snapshot at any time:

```{code-cell}
layout = builder.snapshot()
layout
```

Nested records
--------------

When using a RecordArray form you can not specify a field, the fields alternate.

```{code-cell}
form = """
{
"class": "RecordArray",
"contents": {
    "one": "float64",
    "two": "int64"
},
"form_key": "node0"
}
"""
builder = ak.layout.LayoutBuilder32(form)

# the fields alternate
builder.float64(1.1)  # "one"
builder.int64(2)      # "two"
builder.float64(3.3)  # "one"
builder.int64(4)      # "two"
```

```{code-cell}
layout = builder.snapshot()
layout
```

Similarly, for the record contents with the same type:

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

A more complex example
----------------------

A more complex example that contains both nestsed lists and records:

```{code-cell}
form = """
{
  "class": "ListOffsetArray64",
  "offsets": "i64",
  "content": {
      "class": "RecordArray",
      "contents": {
          "x": {
              "class": "NumpyArray",
              "primitive": "float64",
              "form_key": "node2"
          },
          "y": {
              "class": "ListOffsetArray64",
              "offsets": "i64",
              "content": {
                  "class": "NumpyArray",
                  "primitive": "int64",
                  "form_key": "node4"
              },
              "form_key": "node3"
          }
      },
      "form_key": "node1"
  },
  "form_key": "node0"
}
  """
```

Create a builder from a form:

```{code-cell}
builder = ak.layout.LayoutBuilder32(form)
```

Start appending the data:

```{code-cell}
builder.begin_list()
builder.float64(1.1)
builder.begin_list()
builder.int64(1)
builder.end_list()
builder.float64(2.2)
builder.begin_list()
builder.int64(1)
builder.int64(2)
builder.end_list()
builder.end_list()

builder.begin_list()
builder.end_list()

builder.begin_list()
builder.float64(3.3)
builder.begin_list()
builder.int64(1)
builder.int64(2)
builder.int64(3)
builder.end_list()
builder.end_list()
```

and take a snapshot:

```{code-cell}
layout = builder.snapshot()
layout
```

Error handling
--------------

The commands given to the `LayoutBuilder` must be in the order described by its Form. Issuing a non conforming command or issuing a command in an incorrect order is treated as a user error. As soon as an unexpected command is issued, the builder stops appending data. It is not possible to recover from this state. All you can do is to take a `snapshot` to recover the accumulated data.
