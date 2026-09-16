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

Named axes
==========

Named axes are a feature in Awkward Array that allows you to give names to the axes of an array.
This can be useful for documentation, debugging, and for writing code that is more robust to changes in the structure of the data.
As argumented at [PyHEP.dev 2023](https://indico.cern.ch/event/1234156/) and by the Harvard NLP group in their ["Tensor Considered Harmful"](https://nlp.seas.harvard.edu/NamedTensor.html) write-up, named axes can be a powerful tool to make code more readable and less error-prone.

Awkward array ensures that named axes are properly propagated to the result.
All highlevel, indexing, and broadcasting operations in awkward array support named axes.

Other libraries that support named axes include:
- [hist](https://hist.readthedocs.io/en/latest/)
- [haliax](https://github.com/stanford-crfm/haliax)
- [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor.html)
- [PyTorch Named Tensors](https://pytorch.org/docs/stable/name_inference.html#name-inference-reference-doc)
- [Penzai Named Axis](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html)
- [xarray Named Axis](https://docs.xarray.dev/en/stable/user-guide/indexing.html#)

Named axes in Awkward Array are inspired primarily by `hist` and `PyTorch Named Tensors`.

+++

How to (de-)attach named axes?
-------------------------

Named axes can be attached to an array using the high-level {func}`ak.with_named_axis` function.
Awkward Array allows strings as named axes and integers as positional axes.

The `named_axis` argument of {func}`ak.with_named_axis` accepts either a `tuple` or `dict`:
- `tuple`:
  - `named axis`: item
  - `positional axis`: index of the item
  - _additional_: `None` represents a wildcard for not specifying a name, e.g.: `("x", None)` means that the first axis is named "x" and the second is not named.
- `dict`:
  - `named axis`: key
  - `positional axis`: value
  - _additional_: not specifying a name is not allowed, e.g.: `{"x": 0}` means that the first axis is named "x", all other existing dimensions are unnamed. The `dict` option also allows for renaming negative axes, e.g.: `{"x": -1}` means that the last axis is named "x".


```{code-cell}
import awkward as ak
import numpy as np
```

The axis names of an array can be attached through the constructor:
```{code-cell}
named_array = ak.Array([[1, 2], [3], [], [4, 5, 6]], named_axis=("x", "y"))
# or
named_array = ak.Array([[1, 2], [3], [], [4, 5, 6]], named_axis={"x": 0, "y": 1})
```

... or through `ak.with_named_axis`:
```{code-cell}
array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
named_array = ak.with_named_axis(array, named_axis=("x", "y"))
# or
named_array = ak.with_named_axis(array, named_axis={"x": 0, "y": 1})
```

After attaching named axes, you can see the named axes comma-separated in the arrays representation and in `.show(named_axis=True)`:

```{code-cell}
ak.Array([[1, 2], [3], [], [4, 5, 6]], named_axis=("x", "y"))
```

```{code-cell}
ak.Array([[1, 2], [3], [], [4, 5, 6]], named_axis=("x", "y")).show(named_axis=True)
```

Accessing the named axis mapping to positional axis can be done using the `named_axis` and `positional_axis` properties:

```{code-cell}
named_array.named_axis
```

```{code-cell}
named_array.positional_axis
```

If you want to remove the named axes from an array, you can use the {func}`ak.without_named_axis` function:

```{code-cell}
array = ak.without_named_axis(named_array)
array.named_axis
```


Indexing with Named Axes
------------------------

Named axes can be used for indexing operations.
This is enabled throuhg a special syntax that allows you to index with a dictionary where keys refer to named (or positional) axes and the values to the slice or index.

Simple examples:

```{code-cell}
array = ak.Array([[[1, 2]], [[3]], [[4]], [[5, 6], [7]]])
named_array = ak.with_named_axis(array, named_axis=("x", "y", "z"))

# named axes
named_array[{"x": 0}] # array[0, :, :]
named_array[{"z": 0}] # array[:, :, 0]

named_array[{"x": 0, "y": 0}] # array[0, 0, :]
named_array[{"x": slice(0, 1), "y": 0}] # array[0:1, 0, :]

named_array[named_array > 3] # array[array > 3]


# positional axes
named_array[{0: 0}] # array[0, :, :]
named_array[{2: 0}] # array[:, :, 0]

named_array[{-3: 0}] # array[0, :, :]
named_array[{-1: 0}] # array[:, :, 0]
None
```

If multiple keys that point to the same positional axis are used, the last key will be used and all others will be ignored:

```{code-cell}
array = ak.Array([[[1, 2]], [[3]], [[4]], [[5, 6], [7]]])
named_array = ak.with_named_axis(array, named_axis=("x", "y", "z"))

assert ak.all(named_array[{0: 0, "x": slice(0, 2)}] == named_array[0:2])
assert ak.all(named_array[{"x": slice(0, 2), 0: 0}] == named_array[0])
```


More detailed example:

```{code-cell}
# create a Record Array that represents four events with a variable number of jets
events = ak.zip({
  "event_no": np.arange(4),
  "jetpt": ak.Array([[50, 60], [45], [], [80, 30, 50]]),
})
named_events = ak.with_named_axis(events, ("events", "jets"))

print("classic indexing:", named_events[0, 0:1])
print("named indexing  :", named_events[{"events": 0, "jets": slice(0, 1)}])
```

For syntatic suger, use `np.s_` to define slices more easily:

```{code-cell}
array = ak.Array([[[1, 2]], [[3]], [[4]], [[5, 6], [7]]])
named_array = ak.with_named_axis(array, named_axis=("x", "y", "z"))

assert ak.all(named_array[{"x": np.s_[0:2]}] == named_array[{"x": slice(0, 2)}])
```

Highlevel Operations with Named Axes
------------------------------------

Named axes can be used for specifying the axis of a highlevel operation given that the operation is performed on an array that supports this named axis.

For example, the `ak.sum` operation can be performed on an array with named axes:

```{code-cell}
array = ak.Array([[[1, 2]], [[3]], [[4]], [[5, 6], [7]]])
named_array = ak.with_named_axis(array, named_axis=("x", "y", "z"))

print("Sum over axis 'x':", ak.sum(named_array, axis="x"))  # ak.sum(array, axis=0)
print("Sum over axis 'y':", ak.sum(named_array, axis="y"))  # ak.sum(array, axis=1)
print("Sum over axis 'z':", ak.sum(named_array, axis="z"))  # ak.sum(array, axis=2)
```


Named Axes Propagation Strategies
---------------------------------


Named axes are propagated through all operations in Awkward Array.
For this, specific strategies are defined for each operation to ensure that the named axes are properly propagated to the result.

The possible strategies are:
- `keep all`: keep all named axes
- `keep one`: keep one named axis
- `keep up to`: keep all named axes up to a certain positional axis
- `remove all`: remove all named axis
- `remove one`: remove one named axis
- `add one`: add a new axis
- `unify`: unify named axes of two arrays. The named axes are unifiable if the have the same name (or `None`) and point to the same positional axis.

Indexing operations
:   The following table shows the strategy for indexing operations:

| Operation            | Strategy     |
|----------------------|--------------|
| `array[:]`           | `keep all`   |
| `array[...]`         | `keep all`   |
| `array[()]`          | `keep all`   |
| `array[0:1]`         | `keep all`   |
| `array[[0, 1]]`      | `keep all`   |
| `array[array % 2]`   | `keep all`   |
| `array[0]`           | `remove one` |
| `array[np.array(0)]` | `remove one` |
| `array[None]`        | `add one`    |
| `array[np.newaxis]`  | `add one`    |

Universal functions (`ufuncs`)
:   `ufuncs` with single argument signatures (i.e. unary operations, such as `__abs__`, `__neg__`, `__invert__`, ...) do not modify named axes (strategy: `keep all`).
:   `ufuncs` with two argument signatures (i.e. binary operations, such as `__add__`, `__sub__`, `__mul__`, ...) try to merge named axis of the given arrays (strategy: `unify`).
    This means that the named axes of the two arrays are merged if they have the same name (or either is `None`) and point to the same positional axis.
    If there's a mismatch of named axes, e.g., the same named axis has different names or point to different positional axes, an exception is raised.

```{code-cell}
array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
named_array = ak.with_named_axis(array, named_axis=("x", "y"))

# unary operations with named axes
assert (-named_array).named_axis == {"x": 0, "y": 1}
assert (+named_array).named_axis == {"x": 0, "y": 1}
assert (~named_array).named_axis == {"x": 0, "y": 1}
assert abs(named_array).named_axis == {"x": 0, "y": 1}

# binary operations with named axes
named_array1 = ak.with_named_axis(array, named_axis=(None, "y"))
named_array2 = ak.with_named_axis(array, named_axis=("x", None))
named_array3 = ak.with_named_axis(array, named_axis=("x", "y"))

assert (array + array).named_axis == {}
assert (named_array1 + array).named_axis == {"y": 1}
assert (named_array2 + array).named_axis == {"x": 0}
assert (named_array3 + array).named_axis == {"x": 0, "y": 1}

assert (named_array1 + named_array2).named_axis == {"x": 0, "y": 1}
assert (named_array3 + named_array3).named_axis == {"x": 0, "y": 1}
```

Reducers (`ak.sum`, `ak.any`, ...)
:   If `axis=int` and `keepdims=False` (typical use-case) removes the named axis that is reduced (strategy: `remove one`).
:   If `keepdims=True` is set, the named axis is kept (strategy: `keep all`).
:   If `axis=None` is set, all named axes are removed (strategy: `remove all`).

```{code-cell}
array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
named_array = ak.with_named_axis(array, ("x", "y"))

assert ak.sum(named_array, axis="x", keepdims=False).named_axis == {"y": 0}
assert ak.sum(named_array, axis="x", keepdims=True).named_axis == {"x": 0, "y": 1}
```

---
A full list of operations and their strategies can be found in the following table.
If an operation is not listed, the strategy is either `keep all` or automatically inferred from the below listed operations.


| Operation                                           | Strategy           |
|-----------------------------------------------------|--------------------|
| `ak.all(..., axis=None)`                            | `remove all`       |
| `ak.all(..., axis=int, keepdims=False)`             | `remove one`       |
| `ak.all(..., axis=int, keepdims=True)`              | `keep all`         |
| `ak.any(..., axis=None)`                            | `remove all`       |
| `ak.any(..., axis=int, keepdims=False)`             | `remove one`       |
| `ak.any(..., axis=int, keepdims=True)`              | `keep all`         |
| `ak.[arg]cartesian`                                 | `unify`            |
| `ak.[arg]combinations`                              | `keep all`         |
| `ak.[arg]max(..., axis=None)`                       | `remove all`       |
| `ak.[arg]max(..., axis=int, keepdims=False)`        | `remove one`       |
| `ak.[arg]max(..., axis=int, keepdims=True)`         | `keep all`         |
| `ak.[arg]min(..., axis=None)`                       | `remove all`       |
| `ak.[arg]min(..., axis=int, keepdims=False)`        | `remove one`       |
| `ak.[arg]min(..., axis=int, keepdims=True)`         | `keep all`         |
| `ak.[arg]sort`                                      | `keep all`         |
| `ak.broadcast_arrays`                               | `unify`, `add one` |
| `ak.broadcast_fields`                               | `unify`, `add one` |
| `ak.categories`                                     | `remove all`       |
| `ak.concatenate`                                    | `unify`            |
| `ak.count[_nonzero](..., axis=None)`                | `remove all`       |
| `ak.count[_nonzero](..., axis=int, keepdims=False)` | `remove one`       |
| `ak.count[_nonzero](..., axis=int, keepdims=True)`  | `keep all`         |
| `ak.firsts`                                         | `remove one`       |
| `ak.flatten(..., axis=None)`                        | `remove all`       |
| `ak.flatten(..., axis=0)`                           | `keep all`         |
| `ak.flatten(..., axis=(!=0), keepdims=True)`        | `remove one`       |
| `ak.local_index`                                    | `keep up to`       |
| `ak.num`                                            | `keep one`         |
| `ak.prod(..., axis=None)`                           | `remove all`       |
| `ak.prod(..., axis=int, keepdims=False)`            | `remove one`       |
| `ak.prod(..., axis=int, keepdims=True)`             | `keep all`         |
| `ak.ravel`                                          | `remove all`       |
| `ak.singletons`                                     | `add one`          |
| `ak.sum(..., axis=None)`                            | `remove all`       |
| `ak.sum(..., axis=int, keepdims=False)`             | `remove one`       |
| `ak.sum(..., axis=int, keepdims=True)`              | `keep all`         |
| `ak.unflatten`                                      | `remove all`       |
| `ak.where`                                          | `unify`, `add one` |
| `ak.with_field`                                     | `unify`, `add one` |
| `ak.zip`                                            | `unify`, `add one` |
